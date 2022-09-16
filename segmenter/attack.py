#!/usr/bin/env python3

### update for segmentation model UTnet
### 2022.2.10 

#from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import argparse
import os
import copy
import pandas as pd
import numpy as np
#import tensorflow as tf
import gc
import torch
from torch import nn  
from torchvision.transforms import InterpolationMode
### The following utils and model are adapted from
'''
Gao, Y., Zhou, M., Metaxas, D.N., 2021. Utnet: A hybrid transformer architecture for medical image segmentation, in , Springer. pp. 61–71
Available: https://github.com/yhygao/UTNet
'''
from utils import Data_IO, DeepLearningUtil, Utils, dataLoader, Dice, SegMetricUtils, Preprocess
from utils import Config as config   
from model.UTNet import utnet  
import csv 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PYTORCH_NO_CUDA_MEMORY_CACHING=1
torch.cuda.empty_cache()
#from tensorflow.keras.models import load_model
# import torch ## The model is in torch
#from keras.datasets import cifar10
import pickle
from scipy.optimize import dual_annealing # for local search
import random


# Helper functions
from differential_evolution import differential_evolution
#from scipy.optimize import differential_evolution
import helper

## draw the MRI of attacked img
def plot_attack_MRI(attack_img, img, mask, img_id):
    utils.SaveImageGray([attack_img, img], ['src', 'attacked', 'mask'] + './debug/'+str(img_id)+'/')

class SegmentationPixelAttacker:
    def __init__(self, model, loader, dimensions=(4,256,256), metrics_function=SegMetricUtils.seg_metrics, dice_values_f = Dice.dice_values, least_dice=0.5, sucess_dice=0.5):
        self.model = model[0]
        self.loader = loader
        self.dimensions = dimensions 
        self.metrics_function = metrics_function
        self.least_dice = least_dice ## the threshold (least) dice value of a sample to be attacked
        self.sucess_dice = sucess_dice ## the attacking  sucesses if the its dice <= sucess_dice
        self.dice_values = dice_values_f

    ## get the dice of data with mask
    def model_dice(self, data, mask): 
        pre = self.model(data)
        return self.metrics_function(pre, mask)['dice']

    ## The predict (dice) of MRI image(s) for segmentation
    def predict_dice(self, xs, img, mask, model):  
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        imgs_perturbed = helper.perturb_MRI(xs, img) 
        pre = torch.zeros(len(xs), 2, 256,256, device=config.DEFAULT_DEVICE)
    
        with torch.no_grad():
           i=0
           _batch=100
           while(i < len(xs) // _batch):
               pre[i*_batch:(i+1)*_batch] = model(imgs_perturbed[i*_batch:(i+1)*_batch])
               i = i+1
           if len(xs) % _batch != 0:
               pre[(i-1)*_batch:] = model(imgs_perturbed[(i-1)*_batch:])
        _mask = mask.repeat(len(xs), 1, 1, 1)
        return self.dice_values(pre, _mask).detach().cpu().numpy() 

    def attack_success(self, x, img, mask, model, verbose=False, epsilon=0.5, no_stop=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        # x: an attacking sample, [(pixel),...(pixel)]
        #model.eval()
        imgs_perturbed = helper.perturb_MRI(x, img) 
        pre = model(imgs_perturbed) 
        return True if self.metrics_function(pre, mask)['dice']  <= self.sucess_dice else False 

    def attack(self, img, mask, model, pixel_count=1, img_id=0,
               maxiter=75, popsize=400, verbose=False, plot=False, DE='DE', epsilon=0.5, LS=0, no_stop=False):
        # Change the target class based on whether this is a targeted attack or not 
 
        _min, _max = torch.min(img), torch.max(img) 
        _channel, dim_x, dim_y = self.dimensions 
        bounds =  [(0, dim_x), (0, dim_y), (_min, _max), (_min, _max), (_min, _max), (_min, _max)] * pixel_count ### for MRI

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = popsize ## max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_dice(xs, img, mask, model)

        def callback_fn(x, convergence):
            return self.attack_success(x, img, mask, model, verbose, epsilon=epsilon, no_stop=no_stop)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False, disp=True, DE=DE, LS=LS)
        
        # Calculate some useful statistics to return from this function
        attack_img = helper.perturb_MRI(attack_result.x, img)
        prior_dice = self.model_dice(img, mask)
        attacked_dice = self.model_dice(attack_img, mask)
        cdiff = prior_dice - attacked_dice
        success = (attacked_dice < epsilon)
        # Show the best attempt at a solution (successful or not)
        if plot:
            plot_MRI(attack_img, img, mask, img_id)

        return [model.name, pixel_count, img_id, None, None, success, cdiff, prior_dice,
                attacked_dice, attack_result.x]

    def attack_all(self, models, samples=1, pixels=(1, 3, 5), 
                   maxiter=75, popsize=400, verbose=False, DE='DE', epsilon=0.5, LS=0, no_stop=False, _ids=None, test=True):
        results = []
        ## get the samples to be attacked
        if _ids is None:
            sample_IDs = random.sample(range(len(self.loader)), samples)
        else:
            sample_IDs = _ids.astype(int)
        sample_IDs[0], sample_IDs[1] = 7097,  1016
        sample_IDs.sort()
        for model in models:
            model_results = [] 
            for pixel_count in pixels:                
                #ns = 0
                #for i, batch_datas in enumerate(self.loader): # load a sample
                #    if not (i in sample_IDs): continue
                #    ns = ns + 1
                #    if ns > samples: break
                for i in sample_IDs:
                    batch_datas = self.loader[i]
                    datas, mask = batch_datas
                    img = torch.cat(datas, 1)
                    _dice = self.model_dice(img, mask)
                    print(model.name, '- image', '-', i + 1, '/', len(self.loader), 'dice: ', _dice)
                    #print('The original confidence/dice:  %f' % _dice) 
                    if _dice <= self.least_dice: continue ## do not attack such samples

                    result = self.attack(img, mask, model, pixel_count, img_id = i,
                                         maxiter=maxiter, popsize=popsize,
                                         verbose=verbose, DE=DE, epsilon=epsilon, LS=LS, no_stop=no_stop)
                    model_results.append(result)

            results += model_results
            helper.checkpoint(results)
        return results

        ## in the case of attacking MRI segmentation, if the dice is lower than 50%, then it success.
    def attack_all_least(self, models, samples=1, pixels=(64,), targeted=False, size=(256,256),
                   maxiter=75, popsize=400, verbose=False, DE='DE', epsilon=0.5, LS=0, no_stop=False):
        results = []
        ## get the samples to be attacked
        # sample_IDs = random.sample(range(len(self.loader)), samples)
        # sample_IDs.sort()

        ## just get those samples from ids.npy 
        sample_IDs = np.load('ids.npy', allow_pickle=True)[0:samples]

        for model in models:
            model_results = []
            #for i, batch_datas in enumerate(self.loader): ##i, img in enumerate(img_samples):
            #    if ns > samples: break
            #    if not (i in sample_IDs): continue
            #    ns = ns + 1
            for i in sample_IDs:
                batch_datas = self.loader[i]
                datas, mask = batch_datas
                img = torch.cat(datas, 1)
                _dice = self.model_dice(img, mask)

                print(model.name, '- image',  '-', i , '/', len(self.loader), 'dice: ', _dice) 
                if _dice <= self.least_dice: continue

                ## find the upperbound of pixels that successfully attacks the sample                
                ## THIS IS POSSIBLE OUT OF MEMORY WHEN IT NEEDS MANY PIXELS FOR A SUCESSFUL ATTACK
                ## THE BETTER CHOICE IS SET A UPPER BOUND AS BELOW
                pixel_count = pixels[0]
                print('The number of pixels is:', pixel_count)
                result = self.attack(img, mask, model, pixel_count, img_id = i,
                                   maxiter=maxiter, popsize=popsize,
                                   verbose=verbose, DE=DE, epsilon=epsilon, LS=LS, no_stop=no_stop)
                if not result[5]: ## it fails to attack the sample with the upperbound number of pixels
                    model_results.append(result)
                    continue

                low_pixel_count = 0  
                old_result = copy.copy(result)
                ###pixel_count = pixels[0] # the up_pixel_count
                ##low_pixel_count = 0 ###pixel_count // 2 
                up_pixel_count = pixel_count
                pixel_count = low_pixel_count + (up_pixel_count - low_pixel_count)//2

                ## find the lower bound of number of pixels of successful acctacking the sample
                while(up_pixel_count - low_pixel_count > 1):
                   print('The number of pixels is:', pixel_count)
                   result = self.attack(img, mask, model, pixel_count, img_id = i,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose, DE=DE, epsilon=epsilon, LS=LS, no_stop=no_stop)
                   if len(old_result)==0: 
                       old_result = copy.copy(result)
                       if not result[5]: break # fail with the most pixels

                   if result[5]: # success attack
                       old_result = copy.copy(result)
                       up_pixel_count = pixel_count
                   else:
                       low_pixel_count = pixel_count

                   pixel_count = low_pixel_count + (up_pixel_count - low_pixel_count)//2

                model_results.append(old_result)
            results += model_results
            helper.checkpoint(results, targeted)
        return results
    

 

if __name__ == '__main__':
    '''model_defs = {
        'lenet': LeNet,
        'pure_cnn': PureCnn,
        'net_in_net': NetworkInNetwork,
        'resnet': ResNet,
        'densenet': DenseNet,
        'wide_resnet': WideResNet,
        'capsnet': CapsNet
    }
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
    '''
    _tname=str(os.getpid())
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free > '+_tname)
    memory_gpu=[int(x.split()[2]) for x in open(_tname,'r').readlines()]
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
    os.system('rm '+_tname)
 
    parser = argparse.ArgumentParser(description='Attack models on UTnet') 
    parser.add_argument('--source', default='dr', help='The source of being attacked[cxr/dr/derm].')
    parser.add_argument('--model', default='./model/dr/wb_model.h5', help='The trained model to be attacked.')
    parser.add_argument('--pixels', nargs='+', default=(1,), type=int, help='The number of pixels that can be perturbed.')
    ### e.g., --pixels 1 3
    parser.add_argument('--maxiter', default=35, type=int,
                        help='The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.')
    parser.add_argument('--popsize', default=1000, type=int,
                        help='The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.')
    parser.add_argument('--samples', default=8, type=int, 
                        help='The number of image samples to attack. Images are sampled randomly from the dataset.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--data', default='dr_pos_data_10.npy', help='The data file.')
    parser.add_argument('--type', default=1, type=int, help='The type of attacked samples, 0: negative, 1: positive.')
    parser.add_argument('--LP', action='store_true',  help='Compute the least pixels to sucessfully attack.')
    parser.add_argument('--save', default='results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--DE', default='SHADE', help='The differential evolution algorithm: DE, SHADE, EBLSHADE.')
    parser.add_argument('--epsilon', default=0.5, type=float, help='The confidence threshold.')
    parser.add_argument('--no_stop', action='store_true', help='Do not stop when find a solution.')
    parser.add_argument('--LS', default=0, type=int, help='Do local searching: (0: no local search; 1: local search at the end; 2: local search at each better solution; 4: local search at each better solution and replace the best with the the new one if it is better than the last best one.).')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')
    parser.add_argument('--confidence', default='0.5', help='The least confidence being attack.')

    # The following paremeters are for repeating attack   
    parser.add_argument('--RA', default=1, type=int, help='The repeating times of repeating attacks')
    parser.add_argument('--RAn', default=10, type=int, help='The number samples of repeating attack')
    parser.add_argument('--low', default=0.1, type=float, help='The confidence after first attack')
    parser.add_argument('--up', default=0.9, type=float, help='The confidence before first attack')
    parser.add_argument('--sameIDs', action='store_true', help='The same IDs to be attacked')
    parser.add_argument('--pickle', type=str, help='The results pickle file')

    args = parser.parse_args() 

    ### 2022.2.11 for MRI attack
    ### load the model of UTnet in pytorch 
    model = utnet.UTNet(in_chan = 4, base_chan = 16, num_classes=2)
    device = torch.device(config.DEFAULT_DEVICE)
    model = model.to(device) 
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location = device))
    model.eval()

    ### load the data of BraTS20
    data_set = dataLoader.dataSet([config.DATASET_ROOT], 0)
    data_loader = dataLoader.dataLoader(data_set, 
                                           BatchSize = 1, 
                                           ShuffleFlag = 0, 
										   Size = [256, 256],
                                           ZeroOneFlag = False, 
                                           Patchsize = config.DEFAULT_PATCHSIZE, 
                                           device = config.DEFAULT_DEVICE, 
                                           data_aug=False
                                           )
 
 
    if args.samples == -1 or args.samples > len(data_loader):
       args.samples = len(data_loader)

    attacker = SegmentationPixelAttacker([model], data_loader, least_dice = float(args.confidence))

    print('Starting attack')
   
    if args.LP: ## compute the least number of pixels for a successful attack
        results = attacker.attack_all_least([model], samples=args.samples, pixels=args.pixels, 
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose, DE=args.DE, 
                                  epsilon=args.epsilon, LS=args.LS, no_stop=args.no_stop)
    else:
        if args.sameIDs:
          _ids = np.array(pickle.load(open(r'./results-1/EBL-1-0.1-1000-1000-10.pkl','rb')))[:,2]
        else:
          _ids = None
        results = attacker.attack_all([model], samples=args.samples, pixels=args.pixels, _ids=_ids,
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose, DE=args.DE, 
                                  epsilon=args.epsilon, LS=args.LS, no_stop=args.no_stop)

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs/dice', 'predicted_probs/dice', 'perturbation']
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs/dice']])

    print('Saving to', args.save)
    with open(args.save, 'wb') as file:
        pickle.dump(results, file)
    
    os._exit(0)

    if args.RA > 1: ##重复次数
        _ids = get_IDs_RA(args.up, args.low, 'results/EBL-1-5-0.5-1000-100-1000.pkl', args.type, top_k=args.RAn, segmenation=True)
        # #_test = np.zeros((len(_ids),224,224,3))    
        _test = np.zeros((len(_ids),4,256,256))     ## for segmentation
        for ii in range(min(len(_ids), args.RAn)): 
           print(_ids[ii])
           #_test[i] = test[0][ int(_ids[i][0]) ] ## for classification
           batch_datas = attacker.loader[int(_ids[ii,0])]
           datas, mask = batch_datas
           _test[ii] = torch.cat(datas, 1)[0].cpu().numpy()

        test = _test,  np.array([[args.type]] * len(_ids)) 
        
        #draw the figures with annotated attack points for each sample 
        RA_results = np.array(pickle.load(open(r'results/RA-EBL-0-1-0.1-1000-100-10.pkl','rb')))
        for i in range(min(len(_ids), args.RAn)):
           _path = './results/' + args.source + '/'
           _file_name = _path + 'RA-' +str(i) + '-' + str( int(_ids[i][0]) ) + '.png' 
           
           _prior_dice, _attacked_dice, _cdiff, _nsucc = 0.0, 0.0, 0.0, 0
           fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
           ## draw circles of repeated attacks 
           for j in range(args.RA):
              if  RA_results[j * args.RAn + i][8] > 0.5: continue ## not sucess
              _py, _px = int(RA_results[j * args.RAn + i][9][0]), int(RA_results[j * args.RAn + i][9][1])
              #print(j, _px, _py)
              _i, _j = _py, _px
              _nsucc += 1 
              _cdiff += RA_results[j * args.RAn + i][6]
              _prior_dice +=  RA_results[j * args.RAn + i][7]
              _attacked_dice += RA_results[j * args.RAn + i][8] 
              ax[0,0].scatter(_px, _py, color='', edgecolors=['w'], marker='o', alpha=.5, s=150)
              ax[0,1].scatter(_px, _py, color='', edgecolors=['w'], marker='o', alpha=.5, s=150)
              ax[1,0].scatter(_px, _py, color='', edgecolors=['w'], marker='o', alpha=.5, s=150)
              ax[1,1].scatter(_px, _py, color='', edgecolors=['w'], marker='o', alpha=.5, s=150)
              _test[i][0][_i, _j], _test[i][1][_i, _j], _test[i][2][_i, _j], _test[i][3][_i, _j] \
                      = RA_results[j * args.RAn + i][9][2], RA_results[j * args.RAn + i][9][3], \
                      RA_results[j * args.RAn + i][9][4], RA_results[j * args.RAn + i][9][5]   
           
           rec0 = Preprocess.ZeroOne_Normalization(torch.from_numpy(_test[i, 0, : , :])) * 255    
           rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
           rec1 = Preprocess.ZeroOne_Normalization(torch.from_numpy(_test[i, 1, : , :])) * 255    
           rec1 = rec1.int() ### for numpy.ndarray     astype(np.int64)
           rec2 = Preprocess.ZeroOne_Normalization(torch.from_numpy(_test[i, 2, : , :])) * 255    
           rec2 = rec2.int() ### for numpy.ndarray     astype(np.int64)
           rec3 = Preprocess.ZeroOne_Normalization(torch.from_numpy(_test[i, 3, : , :])) * 255    
           rec3 = rec3.int() ### for numpy.ndarray     astype(np.int64)     
           color = plt.cm.gray
           ax[0,0].imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
           ax[0,1].imshow(rec1.cpu().numpy(), origin='lower', cmap =color)
           ax[1,0].imshow(rec2.cpu().numpy(), origin='lower', cmap =color)
           ax[1,1].imshow(rec3.cpu().numpy(), origin='lower', cmap =color)
           
           fig.suptitle('original dice: ' + format(_prior_dice/_nsucc,'.4f') + ' dice after attack: ' + format(_attacked_dice/_nsucc,'.4f'))       
           #_title = ' cdiff=' + format(_cdiff/_nsucc,'.4f')
           #plt.title(_title)
           plt.xlabel('ID='+str(int(_ids[i][0])))
           #ax.imshow(_test[i], origin='lower')
           plt.savefig(_file_name)
           plt.close('all') 
              
    os._exit(0)          
     

  
