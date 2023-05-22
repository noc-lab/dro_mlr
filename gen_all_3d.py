import os
import random
import shutil
random.seed(3)
import csv
import numpy as np

import scipy.io as io
import sys

import pickle



P_fn='currentrepr_mnist_clean'
N_fn='currentrepr_mnist'
diff_fn='currentreprDiff_mnist'

samples_used=int(float(sys.argv[1]))

hidden_dim=int(float(sys.argv[2]))
seq_len=int(float(sys.argv[3]))

ratio=1

# ----------------------------------------------------- P



def PN_process(filename,target_num):

	random.seed(3)

	########### load data and get sample_dic as before
	data=pickle.load(open(filename+'.pkl','rb'))

  #print(data)
	sample_dic={}
  
	for i,label in enumerate(data['labels']):
		if label not in sample_dic:
			sample_dic[label]=[]

		sample_dic[label].append(data['reprs'][i])
  
	print(sample_dic[0][0].shape)
	print(len(sample_dic[0]))

  
	######### select pairs
  
	n=1
	flag=False
	results=[]
  
	while 1:
		l1=random.randint(0,9)
		l2=random.randint(0,9)

		if l1==l2:
			continue
  	
		ind1=random.randint(0,len(sample_dic[l1])-1)
		ind2=random.randint(0,len(sample_dic[l2])-1)

		ar1=np.array(sample_dic[l1][ind1])
		ar2=np.array(sample_dic[l2][ind2])
   
		if seq_len==1:
			results.append(ar1-ar2)
		else:
			ar=list(ar1-ar2)
			ar=[k for k in ar if np.sum(abs(k))!=0]
			results.extend(ar)

		if len(results)>=target_num:
			break

	results=np.array(results)
	print(results.shape)
	io.savemat(filename+'.mat', {'data': results.astype(float)}) # be careful, must save as float (double), or matlab will fail!!!
	




if seq_len==1:
  target_num=10000
else:
  target_num=100000


PN_process(P_fn,target_num=target_num)
PN_process(N_fn,target_num=target_num)


diff_data=pickle.load(open(diff_fn+'.pkl','rb'))
diff_data=diff_data['reprs']
print(diff_data.shape)
if seq_len==1:
  io.savemat(diff_fn+'.mat', {'data': diff_data.astype(float)})
else:
  diff_data=list(diff_data.reshape(-1,hidden_dim))
  print(len(diff_data))
  diff_data=random.sample(diff_data,int(ratio*len(diff_data)))
  diff_data=np.array(diff_data)
  print(diff_data.shape)
  io.savemat(diff_fn+'.mat', {'data': diff_data.astype(float)})





















