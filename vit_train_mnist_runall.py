import matlab.engine
eng = matlab.engine.start_matlab()
print('matlab engine started')


import os
import random

random.seed(8)

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


DRO_total_stages=10000
epochs_each_DRO_stage='1'

noisy_sample_used='10000'



DRO_target_layer_list=[random.sample(['P','B','0','1','2','3'],1)[0] for i in range(DRO_total_stages)]
#DRO_target_layer_list=([2,4,6,8]*DRO_total_stages)[:DRO_total_stages]
DRO_seed_list=[str(random.randint(1,10000)) for i in range(DRO_total_stages)]
print(DRO_target_layer_list[:50])




DRO_coef={'P':'3e-5',
          'B':'3e-5',
          '0':'5e-6',
          '1':'5e-6',
          '2':'1e-6',
          '3':'5e-6',
          '4':'5e-6',
          '5':'5e-6',
          }

DRO_lr='1e-3'
performance_record_file='results_mnist.csv'


#DRO_target_layer_list=['P']


#DRO_coef={k:'0' for k in DRO_coef} # for baseline methods, set DRO coef to 0

friend_in_DRO='None'



for i,DRO_target_layer in enumerate(DRO_target_layer_list):

  print('Now stage '+str(i))
  
  if os.path.exists('break_flag'):
    print('manually stopped')
    break

  
  #-------------------------- ERM training
  
  '''
  ERM_model='None'
  mode='ERM'
  os.system("python vit_train_mnist_StopGradVer.py %s %s %s %s %s %s %s %s %s %s" % (ERM_model,mode,DRO_coef[DRO_target_layer],DRO_lr,epochs_each_DRO_stage,performance_record_file,DRO_seed_list[i],noisy_sample_used,str(DRO_target_layer),friend_in_DRO))
  
  aa
  '''

  
  # ---------------   generate the repr vectors needed for W estimation
  
  if i==0:
    ERM_model='mnist_vit_ep100.pt'
  else:
    ERM_model='current_mnist_vit.pt'


  #-------------------------- UAP generating
  
  '''
  for u in [0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29]:
  
    mode='UAPgen_'+str(u) # generating UAP using the training set, based on ERM model
    os.system("python vit_train_mnist_StopGradVer.py %s %s %s %s %s %s %s %s %s %s" % (ERM_model,mode,DRO_coef[DRO_target_layer],DRO_lr,epochs_each_DRO_stage,performance_record_file,DRO_seed_list[i],noisy_sample_used,str(DRO_target_layer),friend_in_DRO))
  
  aa'''
  
  
  #-------------------------- 
  
  
  mode='estimate'
  os.system("python vit_train_mnist_StopGradVer.py %s %s %s %s %s %s %s %s %s %s" % (ERM_model,mode,DRO_coef[DRO_target_layer],DRO_lr,epochs_each_DRO_stage,performance_record_file,DRO_seed_list[i],noisy_sample_used,str(DRO_target_layer),friend_in_DRO))
  
  
  
  
  if DRO_target_layer=='P':
    os.system("python gen_all_3d.py %s %s %s" % (noisy_sample_used,'49','16'))
  elif DRO_target_layer=='B':
    os.system("python gen_all_3d.py %s %s %s" % (noisy_sample_used,'64','1'))
  else:
    #os.system("python gen_all_3d.py %s %s %s" % (noisy_sample_used,'72','17')) # one more in seq for CLS token
    os.system("python gen_all_3d.py %s %s %s" % (noisy_sample_used,'64','17')) # qkv version
  
  

  try:
    eng.cvxTC2_sdp(nargout=0)
  except:
    print('matlab crushed, try to restart')
    del(eng)
    eng=0
    eng = matlab.engine.start_matlab() # in case that strange error happens
    eng.cvxTC2_sdp(nargout=0)


  
  if os.path.exists('current_W_.mat'):
    print('matlab running well') # add this check to make sure the previous issue of matlab failing to start won't influence the result
    os.system("mv current_W_.mat current_W.mat")
  else:
    print('MATLAB fails')
    aa
  
  
  
  # ---------------   run DRO 
  

  mode='DRO'
  
  
  os.system("python vit_train_mnist_StopGradVer.py %s %s %s %s %s %s %s %s %s %s" % (ERM_model,mode,DRO_coef[DRO_target_layer],DRO_lr,epochs_each_DRO_stage,performance_record_file,DRO_seed_list[i],noisy_sample_used,str(DRO_target_layer),friend_in_DRO))






















































