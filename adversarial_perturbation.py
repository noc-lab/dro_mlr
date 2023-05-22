import numpy as np
import deepfool
from PIL import Image
#import trainer
import torch
from torchvision import transforms



'''Developed from https://github.com/NetoPedro/Universal-Adversarial-Perturbations-Pytorch, many thanks!'''



def project_perturbation(data_point,p,perturbation  ):

    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation


def generate(trainset, testset, net, noisy_sample_used=10000, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=20,seed=None, image_size=None):
    '''
    :param trainset: Pytorch Dataloader with train data
    :param testset: Pytorch Dataloader with test data
    :param net: Network to be fooled by the adversarial examples
    :param delta: 1-delta represents the fooling_rate, and the objective
    :param max_iter_uni: Maximum number of iterations of the main algorithm
    :param p: Only p==2 or p==infinity are supported
    :param num_class: Number of classes on the dataset
    :param overshoot: Parameter to the Deep_fool algorithm
    :param max_iter_df: Maximum iterations of the deep fool algorithm
    :return: perturbation found (not always the same on every run of the algorithm)
    '''
    
    np.random.seed(seed)
    
    
    net.eval()
    device = 'cuda'

    # Importing images and creating an array with them
    img_trn = []
    for image in trainset:
        #for image2 in image[0]:
        img_trn.append(image.reshape(image_size))
    

    

    img_tst = []
    for image in testset:
        #for image2 in image[0]:
        img_tst.append(image.reshape(image_size))


    # Setting the number of images to 300  (A much lower number than the total number of instances on the training set)
    # To verify the generalization power of the approach
    num_img_trn = noisy_sample_used#100
    index_order = np.arange(num_img_trn)

    # Initializing the perturbation to 0s
    #v=np.zeros([28,28])
    v=0 # this is easier...

    #Initializing fooling rate and iteration count
    fooling_rate = 0.0
    iter = 0

    
    
    fooling_rates=[0]
    accuracies = []
    #accuracies.append(accuracy)
    total_iterations = [0]
    # Begin of the main loop on Universal Adversarial Perturbations algorithm
    
    #while fooling_rate < 1-delta and iter < max_iter_uni:
    while iter < max_iter_uni: # now we only do 1 iter!!!!
        np.random.shuffle(index_order)
        print("Iteration  ", iter)

        for index in index_order:
            
            #v = v.reshape((v.shape[0], -1))

            # Generating the original image from data
            #cur_img = Image.fromarray(img_trn[index][0])
            #cur_img1 = transformer1(transformer2(cur_img))[np.newaxis, :].to(device)
            cur_img=img_trn[index]
            
            #print(index)
            #aa
            
            _, logits, repr_, repr_inViT, raw_patches=net(features=torch.Tensor(cur_img[None,:,:,:]).cuda(), labels=None, DRO_layer='B') # here setting dro layer will not influence anything....
            #print(logits)
            #print(logits.shape)
            r2 = torch.argmax(logits.flatten(), dim=-1).cpu().numpy()
            #print(r2)

            # Feeding the original image to the network and storing the label returned
            #r2 = (net(cur_img1).max(1)[1])
            torch.cuda.empty_cache()


            # Generating a perturbed image from the current perturbation v and the original image
            #per_img = Image.fromarray(transformer2(cur_img)+v.astype(np.uint8))
            per_img = cur_img+v
            #per_img1 = transformer1(transformer2(per_img))[np.newaxis, :].to(device)

            # Feeding the perturbed image to the network and storing the label returned
            _, logits, repr_, repr_inViT, raw_patches=net(features=torch.Tensor(per_img[None,:,:,:]).cuda(), labels=None, DRO_layer='B')
            #r1 = (net(per_img1).max(1)[1])
            r1 = torch.argmax(logits.flatten(), dim=-1).cpu().numpy()
            torch.cuda.empty_cache()
            
            

            # If the label of both images is the same, the perturbation v needs to be updated
            if r1 == r2:
                print(">> k =", np.where(index==index_order)[0][0], ', pass #', iter)

                # Finding a new minimal perturbation with deepfool to fool the network on this image
                #dr, iter_k, label, k_i, pert_image = deepfool.deepfool(per_img1[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                dr, iter_k, label, k_i, pert_image = deepfool.deepfool(image=per_img, 
                                                                        net=net, 
                                                                        num_classes=num_classes, 
                                                                        overshoot=overshoot, 
                                                                        max_iter=max_iter_df,
                                                                        )
                #print(dr.shape)
                #aa
                
                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df-1:

                    #v[:, :] += dr[0,0, :, :]
                    v += dr # this is easier...

                    v = project_perturbation( xi, p,v)

        iter = iter + 1


    return v
