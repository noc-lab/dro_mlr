
## This file source is https://github.com/BXuan694/Universal-Adversarial-Perturbation/blob/master/deepfool.py
## This file is not the scope of the original paper of this project

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy


#from torch.autograd.gradcheck import zero_gradients # this is very old and not available in torch 1.11

def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()


def deepfool(image, net, num_classes, overshoot, max_iter):

    """
       :param image:
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        #image = image.cuda()
        net = net.cuda()

    _, logits, repr_, repr_inViT, raw_patches = net.forward(features=Variable(torch.Tensor(image[None, :, :, :]).cuda(), requires_grad=True),
                                                            labels=None,
                                                            DRO_layer='B',
                                                            )
    f_image=logits.data.cpu().numpy().flatten()
    
    #print(f_image)
    #print(f_image.shape)
    #aa
    
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]
    
    #print(label)
    

    #input_shape = image.cpu().numpy().shape
    input_shape = image.shape
    #print(input_shape)
    #aa
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(torch.Tensor(pert_image[None, :,:,:]).cuda(), requires_grad=True)
    
    #fs = net.forward(x)
    _, fs, repr_, repr_inViT, raw_patches = net.forward(features=x,
                                                            labels=None,
                                                            DRO_layer='B',
                                                            )
    
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            #zero_gradients(x)
            x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            
            #print(cur_grad.shape)
            #aa

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        
        #r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_i =  (pert+1e-4) * w[0,:,:,:] / np.linalg.norm(w) # first dim is batchsize, which is 1, we drop it
        

        
        r_tot = np.float32(r_tot + r_i)

  
        pert_image=image+(1+overshoot)*r_tot

        
        
        
        x = Variable(torch.Tensor(pert_image[None, :,:,:]).cuda(), requires_grad=True)
       # print(image.shape)
       # print(x.view(1,1,image.shape[0],-1).shape)
        #fs = net.forward(x.view(1,1,image.shape[1],-1))
        _, fs, repr_, repr_inViT, raw_patches = net.forward(features=x,
                                                            labels=None,
                                                            DRO_layer='B',
                                                            )
        

        
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image













