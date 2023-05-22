import os
import sys

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import numpy as np
#from skimage.io import imread,imsave
import random
seed=3
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

import csv
import copy
import pandas as pd
from scipy.linalg import sqrtm
import pickle

from einops.layers.torch import Rearrange, Reduce

from transformers import ViTFeatureExtractor, ViTModel, ViTConfig

from transformers.models.vit.modeling_vit import ViTEncoder, ViTLayer, ViTEmbeddings
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput





pair = lambda x: x if isinstance(x, tuple) else (x, x)






our_config={
        'eval_epochs':1,
        'save_epochs':10,#100,
        'training_epoch':100,
        'batchsize':1000,
        'eval_batchsize':1000,
        'wd':0.01,
        'lr':1e-3,
        'ckpt': None,
        
        'current_model':sys.argv[1],
        
        'is_training':sys.argv[2]=='ERM',
        'is_predict':sys.argv[2]=='estimate',
        'is_gen_repr_csv':sys.argv[2]=='estimate',
        'is_DRO_training':sys.argv[2]=='DRO',
        'is_adv_training':sys.argv[2]=='AT',
        'is_UAP_generating':sys.argv[2].find('UAPgen')!=-1,#sys.argv[2]=='UAPgen',
        
        'DRO_coef':float(sys.argv[3]),
        #'DRO_coef':2e-6,
        
        'DRO_lr':float(sys.argv[4]),
        
        'epochs_each_DRO_stage':int(sys.argv[5]),
        'performance_record_file':sys.argv[6],
        
        'DRO_seed':int(float(sys.argv[7])),
        'noisy_sample_used':int(float(sys.argv[8])),
        'DRO_target_layer':int(float(sys.argv[9])) if sys.argv[9] not in ['P','B'] else sys.argv[9],
        
        'friend_in_DRO':sys.argv[10], # 'AT' or 'PGD' or 'None'
        
        }


#print(random.randint(1,10000))
#aa


# ----------------------------------- load and split the images



def load_and_split(filename):
  
  #random.seed(seed)
  dataset_=[]
  #lines=[item for item in csv.reader(open(filename, "r",encoding='utf-8'))]
  lines=np.load(filename)

  
  train=lines[:50000,:]
  val=lines[50000:60000,:]
  test=lines[60000:,:]

  return train, val, test




# -------------------------------------- build the model



import collections.abc

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class our_PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches


        #image_height, image_width = pair(image_size)
        #patch_height, patch_width = pair(patch_size)
        
        self.rearrange=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1])
        patch_dim = num_channels * patch_size[0] * patch_size[1]
        self.our_projection=nn.Linear(patch_dim, embed_dim) # replace the convolution type of projection layer into a simple linear layer to apply DRO

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        
        
        x=self.rearrange(pixel_values) # replace the conv to linear layer, so later we can do DRO
        patches_before_linear=x
        x=self.our_projection(x)
        
        #x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        
        #print(x.shape)
        return x, patches_before_linear




class our_ViTEmbeddings(ViTEmbeddings):

    def __init__(self, config):
        super().__init__(config)


        self.patch_embeddings = our_PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        #num_patches = self.patch_embeddings.num_patches


    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings, patches_before_linear = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, patches_before_linear





class our_ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, input_tensor):
    
        hidden_states_before_linear=hidden_states
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + input_tensor
        
        return hidden_states, hidden_states_before_linear





class our_ViTLayer(ViTLayer):
  
    def __init__(self, config):
        super().__init__(config)
        
        self.output = our_ViTOutput(config)
        

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        
        hidden_states_before_qkv=self.layernorm_before(hidden_states) # extract this for DRO
        self_attention_outputs = self.attention(
            hidden_states_before_qkv,#self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        # TODO feedforward chunking not working for now
        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layer_output
        # )

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        #layer_output = self.output(layer_output, hidden_states)
        layer_output, hidden_states_before_linear = self.output(layer_output, hidden_states)

        #outputs = (layer_output,) + outputs + (hidden_states_before_linear,)
        outputs = (layer_output,) + outputs + (hidden_states_before_qkv, hidden_states_before_linear)

        return outputs



class our_ViTEncoder(ViTEncoder):
    
    def __init__(self, config):
        super().__init__(config)

        self.layer = nn.ModuleList([our_ViTLayer(config) for _ in range(config.num_hidden_layers)])


    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        DRO_layer=None, # assign which layer shall we do DRO, so the corresponding repr vectors will be returned
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            
            if DRO_layer!=None:
                if i==DRO_layer:
                    hidden_states_before_linear=layer_outputs[-1] # added to get repr vectors for DRO
                    hidden_states_before_qkv=layer_outputs[-2] # added to get repr vectors for DRO
            else:
                hidden_states_before_linear=None
                hidden_states_before_qkv=None


            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            #hidden_states=all_hidden_states,
            #hidden_states=hidden_states_before_linear, # now we use this to pass the repr vectors
            hidden_states=hidden_states_before_qkv, # now we try DRO for qkv linear layer instead of MLP layer
            attentions=all_self_attentions,
        )



class our_ViTModel(ViTModel):

    def __init__(self, config):
        super().__init__(config)

        self.encoder = our_ViTEncoder(config)
        self.embeddings = our_ViTEmbeddings(config)

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        DRO_layer=None,
    ):


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, patches_before_linear = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            DRO_layer=DRO_layer,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=patches_before_linear,#pooled_output, # now we use this to pass the patches!!
            hidden_states=encoder_outputs.hidden_states, # now we use this to pass the repr vectors!!
            attentions=encoder_outputs.attentions,
        )




class vision_model(nn.Module):
  
  def __init__(self,config):
  
    super().__init__()
    
    self.config = config
    #self.vit=ViTModel(self.config)
    self.vit=our_ViTModel(self.config) 
    
    self.B= nn.Linear(config.hidden_size, 10)



  def forward(
      self,
      features=None,
      labels=None,
      output_attentions=False,
      DRO_layer=None,
  ):


    if DRO_layer in ['P','B']:
      DRO_layer=0 # temporarily set to 0. we will not use this at all anyway
    _out=self.vit(features,output_attentions=output_attentions,DRO_layer=DRO_layer)
    vit_outputs = _out.last_hidden_state
    cls_output=vit_outputs[:,0,:]
    
    hidden_states_before_linear=_out.hidden_states
    patches_before_linear=_out.pooler_output
    #print(cls_output.shape)
    
    repr_=cls_output
    
    logits = self.B(repr_)

    #print(repr_.shape)
    
    loss=None
    
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, 10), labels.view(-1))
        
    
    return loss, logits, repr_, hidden_states_before_linear, patches_before_linear





# --------------------------------- model training/test




def pred_one_dataset_batch(model,dataset,batchsize=our_config['eval_batchsize'],output_repr=False,DRO_layer=None):


  model.eval()
  PRED=[]
  AUC=None
  ACC=None
  REPR=[]
  REPR_inViT=[]
  PATCHES=[]
  
  LOSS=[]

  for r in range(int(len(dataset)/batchsize)+1):
    
    #print(r)
    
    eval_index=[i for i in range(len(dataset))]
    ind_slice=eval_index[r*batchsize:(r+1)*batchsize]
    
    if ind_slice==[]:
      continue
    

    X=dataset[ind_slice,1:]
    y=dataset[ind_slice,0]



    X=torch.Tensor(X).to('cuda')
    X=torch.reshape(X,(-1,1,28,28))

    X=X/255

    y=torch.LongTensor(y).to('cuda')


    _, output, repr_, repr_inViT, raw_patches=model(features=X, labels=y, DRO_layer=DRO_layer)
    

    pred=torch.argmax(output, dim=-1).cpu().numpy().tolist()
    repr_inViT=repr_inViT.cpu().detach().numpy()#.tolist()
    raw_patches=raw_patches.cpu().detach().numpy()#.tolist()



    repr_=repr_.cpu().detach().numpy()#.tolist()
    PRED.extend(pred)
    REPR.extend(repr_)
    REPR_inViT.extend(repr_inViT)
    PATCHES.extend(raw_patches)

    
    loss=_.cpu().detach().numpy()
    #print(loss) # here loss is a number
    #print(loss.shape)
    
    LOSS.append(loss*len(ind_slice))
  LOSS=np.sum(LOSS)/len(dataset)

  

  GT=dataset[:,0].astype(np.uint8).tolist()
  
  #print(GT[:50])
  #print(PRED[:50])
  
  assert(len(PRED)==len(GT))
  
  ACC=np.mean([GT[i]==PRED[i] for i in range(len(GT))])


  if not output_repr:
    return PRED, LOSS, ACC
  else:
    return PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES




from torch.autograd import Variable

def LinfPGDAttack(model,features,labels,epsilon=0.2,k=5,a=0.1):
  
  features_nat=copy.deepcopy(features)
  
  for i in range(k):
  
    features_=Variable(copy.deepcopy(features),requires_grad=True)
    labels_=copy.deepcopy(labels)
    

    loss, output, repr_, repr_inViT, raw_patches = model(features=features_, labels=labels_, DRO_layer=our_config['DRO_target_layer'])
    
    #print(features_.grad)
    loss.backward()
    gradient=features_.grad
    #print(features_.grad)
    #print(features_.grad.shape)
    
    features_tp=features+a*torch.sign(gradient)
    features_tp=torch.clamp(features_tp,min=features_nat-epsilon,max=features_nat+epsilon)
    features_tp=torch.clamp(features_tp,min=0,max=1)
    
    features=features_tp
    
    features_.grad.zero_()
    
  return features




data_folder='/xxx/'



if our_config['is_training']==True:

  train,val,test=load_and_split(filename=data_folder+'mnist_GW_0.npy')
  
  print(train.shape)
  print(val.shape)
  print(test.shape)

  config=ViTConfig.from_pretrained("vit_config/")
  model=vision_model(config).cuda()

  
  optimizer = torch.optim.AdamW(model.parameters(), lr=our_config['lr'], weight_decay=our_config['wd'])

  
  batchsize=our_config['batchsize']
  all_index=[i for i in range(len(train))]
  random.seed(seed)
  
  
  for e in range(1,our_config['training_epoch']+1): 
      
      if our_config['is_training']==False:
          break
      
      # training for each epoch -----------------------------------
      
      model.train()
      
      random.shuffle(all_index)
      for r in range(int(len(train)/batchsize)): # no +1: in training, make sure the batchsize is stable
          
          ind_slice=all_index[r*batchsize:(r+1)*batchsize]

          X=train[ind_slice,1:]
          y=train[ind_slice,0]

  
          X=torch.Tensor(X).to('cuda')
          X=torch.reshape(X,(-1,1,28,28))
          X=X/255
          y=torch.LongTensor(y).to('cuda')
          
          optimizer.zero_grad()
          loss, output, repr_, repr_inViT, raw_patches = model(features=X, labels=y, DRO_layer=our_config['DRO_target_layer'])
          l_numerical = loss.item()
          
          loss.backward()
          optimizer.step()
  
      print(f"Epoch: {e}, Loss: {l_numerical}")
      
      
      if e%our_config['eval_epochs']==0:
          #continue
        
        PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=val,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
        print('ep'+str(e)+' val LOSS: ',LOSS)
        print('ep'+str(e)+' val ACC: ',ACC)
  
        PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=test,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
        print('ep'+str(e)+' test LOSS: ',LOSS)
        print('ep'+str(e)+' test ACC: ',ACC)
      
      
      if e%our_config['save_epochs']==0:
        torch.save(model.state_dict(), 'mnist_vit_ep'+str(e)+'.pt')




def write_repr_csv(filename,repr_vectors,labels,dim=2):
  
  out = open(filename, 'a', newline='',encoding='utf-8')
  csv_write = csv.writer(out, dialect='excel')

  assert(len(repr_vectors)==len(labels))
	
  for i in range(len(labels)):
    
    if dim==2:
      csv_write.writerow([labels[i]]+repr_vectors[i].tolist())
    elif dim==3:
      flattened_vectors=[]
      for vec in repr_vectors[i].tolist():
        flattened_vectors.extend(vec)
      csv_write.writerow([labels[i]]+flattened_vectors)
    else:
      print('not implemented')
      aa
	
  out.close()



############# filename list




filename_list=[data_folder+'mnist_GW_0.npy',
                data_folder+'mnist_GW_0.1.npy',
                data_folder+'mnist_GW_0.18.npy',
                data_folder+'mnist_GW_0.2.npy',
                data_folder+'mnist_GW_0.3.npy',
                data_folder+'mnist_GW_0.4.npy',
                data_folder+'mnist_GW_0.5.npy',
                data_folder+'mnist_GW_0.6.npy',

                ]

filename_for_repr=data_folder+'mnist_GW_0.18.npy'


if our_config['is_predict']==True:
  

  config=ViTConfig.from_pretrained("vit_config/")
  model=vision_model(config).cuda()
  model.load_state_dict(torch.load(our_config['current_model']))

  

  model.eval()


  #if our_config['is_gen_repr_csv']==False:
  
  for filename in filename_list:
    
    print('Now testing the test set in '+filename)
    noisy_train,noisy_val,noisy_test=load_and_split(filename=filename)
    #PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_val,output_repr=True, DRO_layer=our_config['DRO_target_layer'])
    PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_test,output_repr=True, DRO_layer=our_config['DRO_target_layer'])
    print('test LOSS: ',LOSS)
    print('test ACC: ',ACC)
    

  
  

  print('Now testing the val set in '+data_folder+'mnist_GW_0.npy')
  train,val,test=load_and_split(filename=data_folder+'mnist_GW_0.npy')
  #PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=val,output_repr=True, DRO_layer=our_config['DRO_target_layer'])
  PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=train[:our_config['noisy_sample_used'],:],output_repr=True, DRO_layer=our_config['DRO_target_layer'])
  print('val LOSS: ',LOSS)
  print('val ACC: ',ACC)
  
  

  if our_config['is_gen_repr_csv']==True:
    if our_config['DRO_target_layer']=='P':
      pickle.dump({'reprs':np.array(PATCHES), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist_clean.pkl','wb'))
    elif our_config['DRO_target_layer']=='B':
      pickle.dump({'reprs':np.array(REPR), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist_clean.pkl','wb'))
    else:
      pickle.dump({'reprs':np.array(REPR_inViT), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist_clean.pkl','wb'))



  
  print('Now testing the val set in '+filename_for_repr)
  noisy_train,noisy_val,noisy_test=load_and_split(filename=filename_for_repr)  
  #PRED, LOSS, ACC, REPR_noisy, REPR_inViT_noisy, PATCHES_noisy=pred_one_dataset_batch(model,dataset=val,output_repr=True, DRO_layer=our_config['DRO_target_layer'])
  PRED, LOSS, ACC, REPR_noisy, REPR_inViT_noisy, PATCHES_noisy=pred_one_dataset_batch(model,dataset=noisy_train[:our_config['noisy_sample_used'],:],output_repr=True, DRO_layer=our_config['DRO_target_layer'])
  print('val LOSS: ',LOSS)
  print('val ACC: ',ACC)




  if our_config['is_gen_repr_csv']==True:
    if our_config['DRO_target_layer']=='P':
      pickle.dump({'reprs':np.array(PATCHES_noisy), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist.pkl','wb'))
    elif our_config['DRO_target_layer']=='B':
      pickle.dump({'reprs':np.array(REPR_noisy), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist.pkl','wb'))
    else:
      pickle.dump({'reprs':np.array(REPR_inViT_noisy), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist.pkl','wb'))





  if our_config['is_gen_repr_csv']==True:
    if our_config['DRO_target_layer']=='P':
      pickle.dump({'reprs':np.array(PATCHES_noisy)-np.array(PATCHES), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentreprDiff_mnist.pkl','wb'))
    elif our_config['DRO_target_layer']=='B':
      pickle.dump({'reprs':np.array(REPR_noisy)-np.array(REPR), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentreprDiff_mnist.pkl','wb'))
    else:
      pickle.dump({'reprs':np.array(REPR_inViT_noisy)-np.array(REPR_inViT), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentreprDiff_mnist.pkl','wb'))









# ---------------- UAP generating


from adversarial_perturbation import generate





if our_config['is_UAP_generating']==True:

  train,val,test=load_and_split(filename=data_folder+'mnist_GW_0.npy')
  
  print(train.shape)
  print(val.shape)
  print(test.shape)
  #print(train[34])
  #aa

  config=ViTConfig.from_pretrained("vit_config/")
  model=vision_model(config).cuda()
  model.load_state_dict(torch.load(our_config['current_model']))
  model.eval()
  
  
  #xi=0.11
  xi=float(sys.argv[2].split('_')[-1])
  print('UAP xi:',xi)
  
  v = generate(trainset=(train[:,1:].astype(float))/255,
                                                          testset=(test[:,1:].astype(float))/255, 
                                                          noisy_sample_used=10000,
                                                          net=model, 
                                                          delta=0.2, 
                                                          max_iter_uni=1,#np.inf,
                                                          xi=xi, 
                                                          p=np.inf, 
                                                          num_classes=10, 
                                                          overshoot=0.2, 
                                                          max_iter_df=20,
                                                          seed=int(xi*1000),
                                                          image_size=(1,28,28),
                                                          )
  
  print(v.reshape(784))
  print(v.shape)
  
  
  test=test.astype(float)
  test[:,1:]=test[:,1:]/255+v.reshape(1,784)
  test[:,1:]=np.clip(test[:,1:],0,1)
  test[:,1:]=test[:,1:]*255
  test=test.astype(np.uint8)
  print(test)
  print(test.shape)
  
  #from PIL import Image
  #image=Image.fromarray(test[1,1:].reshape(28,28).astype(np.uint8))
  #image.show()
  
  PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=test,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
  print('test LOSS: ',LOSS)
  print('test ACC: ',ACC)
  
  
  # now save the UAP and perturbed images
  
  data=np.load(data_folder+'mnist_GW_0.npy')

  data=data.astype(float)
  data[:,1:]=data[:,1:]/255+v.reshape(1,784)
  data[:,1:]=np.clip(data[:,1:],0,1)
  data[:,1:]=data[:,1:]*255
  data=data.astype(np.uint8)

  print(data)
  print(data.shape)

  np.save(data_folder+'mnist_UAP_'+str(xi)+'.npy', data)
  np.save(data_folder+'mnist_UAP_'+str(xi)+'_UAPtensor.npy', v)
  
  aa





# ---------------- DRO training


class DRO(nn.Module):
  
  def __init__(self,config):
  
    super().__init__()
    
    self.config = config
    #self.vit=ViTModel(self.config)
    self.vit=our_ViTModel(self.config) 
    

    self.B= nn.Linear(config.hidden_size, 10)


  def forward(
      self,
      features=None,
      labels=None,
      output_attentions=False,
      DRO_layer=None,
      W_minus_half=None,
      DRO_coef=None,
  ):
  


    _out=self.vit(features,output_attentions=output_attentions,DRO_layer=DRO_layer)
    vit_outputs = _out.last_hidden_state
    cls_output=vit_outputs[:,0,:]
    
    hidden_states_before_linear=_out.hidden_states
    patches_before_linear=_out.pooler_output
    #print(cls_output.shape)
    
    repr_=cls_output
    
    output = self.B(repr_)

    #print(repr_.shape)
    loss=None
    
    if labels is not None:

      loss_fct = nn.CrossEntropyLoss()


      if our_config['DRO_target_layer']=='P':
        for W in self.vit.embeddings.patch_embeddings.our_projection.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)      
            r = torch.max(S)

      elif our_config['DRO_target_layer']=='B':
        for W in self.B.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)     
            r = torch.max(S)
      
      else:
        '''for W in self.vit.encoder.layer[our_config['DRO_target_layer']].output.dense.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r = torch.max(S)'''

        for W in self.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.query.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r1 = torch.max(S)

        for W in self.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.key.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r2 = torch.max(S)

        for W in self.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.value.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r3 = torch.max(S)
   
        r=r1+r2+r3
        
      loss = loss_fct(output.view(-1, 10), labels.view(-1))+DRO_coef*r


    return loss, output


import scipy.io


if our_config['is_DRO_training']==True:


  train,val,test=load_and_split(filename=data_folder+'mnist_GW_0.npy')
  
  print(train.shape)
  print(val.shape)
  print(test.shape)


  if our_config['friend_in_DRO']=='AT':
    noisy_train,noisy_val,noisy_test=load_and_split(filename=filename_for_repr)
    train=np.concatenate((train,noisy_train[:our_config['noisy_sample_used'],:]), axis=0)
  

  W=scipy.io.loadmat('current_W.mat')

  W=W['W']
  
  W_inv=np.linalg.inv(W)
  W_minus_half=sqrtm(W_inv)
  W_half=sqrtm(W)


  W=W.astype(np.float32)
  W_inv=W_inv.astype(np.float32)
  W_minus_half=W_minus_half.astype(np.float32)
  W_half=W_half.astype(np.float32)
  
  W_minus_half=torch.Tensor(W_minus_half).cuda()
  


  config=ViTConfig.from_pretrained("vit_config/")


  model=vision_model(config).cuda()
  model.load_state_dict(torch.load(our_config['current_model']))
  model.eval()
  

  DRO_trainer=DRO(config).cuda()
  DRO_trainer.load_state_dict(model.state_dict(),strict=True) # directly load all layers
  DRO_trainer.train()


  
  for param in DRO_trainer.parameters():
    param.requires_grad = False

  
  if our_config['DRO_target_layer']=='P':
    for param in DRO_trainer.vit.embeddings.patch_embeddings.our_projection.parameters():
      param.requires_grad = True
  elif our_config['DRO_target_layer']=='B':
    for param in DRO_trainer.B.parameters():
      param.requires_grad = True
  else:
    #for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].output.dense.parameters():
      #param.requires_grad = True
    for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.query.parameters():
      param.requires_grad = True
    for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.key.parameters():
      param.requires_grad = True
    for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.value.parameters():
      param.requires_grad = True

  '''
  for param in DRO_trainer.vit.embeddings.patch_embeddings.our_projection.parameters():
    param.requires_grad = True
  for param in DRO_trainer.B.parameters():
    param.requires_grad = True
  for tl in [0,1,2,3]:
    for param in DRO_trainer.vit.encoder.layer[tl].attention.attention.query.parameters():
      param.requires_grad = True
    for param in DRO_trainer.vit.encoder.layer[tl].attention.attention.key.parameters():
      param.requires_grad = True
    for param in DRO_trainer.vit.encoder.layer[tl].attention.attention.value.parameters():
      param.requires_grad = True'''


  optimizer = torch.optim.Adam(DRO_trainer.parameters(), lr=our_config['DRO_lr'])
  
  batchsize=our_config['batchsize']
  all_index=[i for i in range(len(train))]
  random.seed(our_config['DRO_seed'])
  
  for e in range(1,our_config['epochs_each_DRO_stage']+1): 
      
  

      random.shuffle(all_index)
      
      #print(DRO_trainer.fc3.state_dict())
      
      
      for r in range(int(len(train)/batchsize)):

   
          ind_slice=all_index[r*batchsize:(r+1)*batchsize]


          X=train[ind_slice,1:]
          y=train[ind_slice,0]


          X=torch.Tensor(X).to('cuda')
          X=torch.reshape(X,(-1,1,28,28))
          X=X/255
          y=torch.LongTensor(y).to('cuda')
          
          optimizer.zero_grad()
          
          if our_config['friend_in_DRO']=='PGD':
            X=LinfPGDAttack(model=model,features=X,labels=y,epsilon=0.06,k=5,a=0.02)
          
          loss, output = DRO_trainer(features=X, labels=y, W_minus_half=W_minus_half, DRO_coef=our_config['DRO_coef'])
          l_numerical = loss.item()
          
          loss.backward()
          optimizer.step()
          
          model.load_state_dict(DRO_trainer.state_dict(),strict=True)
          
      print(f"Epoch: {e}, Loss: {l_numerical}")
      
      


  model.load_state_dict(DRO_trainer.state_dict(),strict=True) # directly load all layers

  
  
  # ------------------ test DRO model
  



  all_ACC=[]
  all_LOSS=[]
  
  for filename in filename_list:
    
    print('Now testing the test set in '+filename)
    noisy_train,noisy_val,noisy_test=load_and_split(filename=filename)
    PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_test,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
    #PRED, LOSS, ACC, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_val,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
    print('test LOSS: ',LOSS)
    print('test ACC: ',ACC)
    
    all_ACC.append(ACC)
    all_LOSS.append(LOSS)

  out = open(our_config['performance_record_file'], 'a', newline='',encoding='utf-8')
  csv_write = csv.writer(out, dialect='excel')
  csv_write.writerow(sys.argv+[e,l_numerical]+all_ACC+all_LOSS)
  out.close()
  

  torch.save(model.state_dict(), 'current_mnist_vit.pt')
  



