import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils.Utils import SaveImageGray
from utils import dataLoader
from collections import Counter
from utils import Preprocess, dataLoader, Dice
import torch 
from helper import perturb_MRI
from utils import Config as config
from model.UTNet import utnet



def read_st(fn, n=10, npixels=1): ## just compute 10 samples 
# fn: file name
# n: number of samples
# read the data from the file fn and return the data 
   sn = n  

   Data = [] # for all samples
   f = open(fn,"r")

   # skip the two lines:
   # Evaluating model_1
   # Starting attack
   #line = f.readline()
   line = f.readline()
   line = f.readline().split() 
   i=0
   while i<sn:
     while (float(line[8]) < 0.9):
        line = f.readline().split()
        while(len(line)!=9): 
           line = f.readline().split() 
     data = [] # for one sample
     data.append( float(line[4] ) )
     data.append( float(line[8] ) ) 
     line = f.readline().split()
     while(len(line) == 5):
        data.append( float(line[4]) )
        line = f.readline().split()
     # skip the line 
     # The minimal number of successful attacking pixels is  
     Data.append(data)
     i+=1 

   fig, ax = plt.subplots() 
   cdiff,wdiff = 0.0, 0.0
   for j in range(len(Data)):
      plt.plot(Data[j][2:], label=str(int(Data[j][0]))+': '+format(Data[j][1],'.4f')) 
      cdiff += (Data[j][1]-Data[j][len(Data[j])-1])
      wdiff += Data[j][1]*(Data[j][1]-Data[j][len(Data[j])-1])

   fig.subplots_adjust(right=0.7, bottom=0.2)
   plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0) #loc='lower right',mode='expand')
   plt.xlabel('iterations' + '\ncdiff='+format(cdiff/len(Data),'.4f') + ' wdiff=' + format(wdiff/len(Data),'.4f'))
   plt.ylabel('dice after attack') 
   plt.savefig('results/'+str(npixels) + '-1000-100-10.png')
   plt.cla()
   plt.clf()
   plt.close('all')
   return Data 
 

def draw_1000s(post_fn='-0-1-0.01-500-1000-10-1'):
  # get the Data
  types = ['DE', 'SHA', 'EBL']
  path='./results/dr/' 
  DATA = []
  TIME = []
  _data,_time = read_st(path + types[i] + post_fn + '.txt',50)
  DATA.append(_data)
  TIME.append(_time)

  # print the original confidence
  for i in range(len(DATA[0])):
     print('ID=',i, '  dice=', DATA[0][i][1])
  #return
  # draw DE, SHA, EBL
  for i in range(1):
     data = np.array(DATA[i])  
     fig, ax = plt.subplots() 
     cdiff,wdiff = 0.0, 0.0
     for j in range(len(data)):
        plt.plot(data[j][2:], label=str(j)+': '+format(data[j][1],'.4f')) 
        cdiff += (data[j][1]-data[j][len(data[j])-1])
        wdiff += data[j][1]*(data[j][1]-data[j][len(data[j])-1])

     fig.subplots_adjust(right=0.7, bottom=0.2)
     plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)  
     plt.xlabel('iterations' + '\ncdiff='+format(cdiff/len(data),'.4f') + ' wdiff=' + format(wdiff/len(data),'.4f'))
     plt.ylabel('confidence')
     plt.title(types[i] + post_fn + '-' + TIME[i])
     plt.savefig(path+types[i]+post_fn+'.png')
     plt.cla()
     plt.clf()
     plt.close('all')

  
  # draw the confidence for each sample with DE, SHA and EBL
  for i in range(len(DATA[0])):
     plt.figure(clear=True)
     X = np.linspace(1,1000,1000,endpoint=True)
     plt.title('ID=' + str(i) +'  Confidence='+ format(DATA[0][i][1],'.5f') )
     for i in range(3):
        plt.plot(X, DATA[j][i][2:], label=types[j])
     plt.legend()
     plt.savefig(path + 'id-' + str(i) + post_fn + '.png')

# draw the scater of positive and negative samples
def draw_1000_scater(path='./results/dr/', post_fn='',le=0):
  # get the Data
  types = ['EBL'] 
  DATA = []
  TIME = []
  label = ['negative', 'positive']
  for i in range(2):
     _data,_time = read_st(path + types[0] + post_fn + str(i)+'.txt',1000)
     DATA.append(_data)
     TIME.append(_time)
  
  cdiff,wdiff = 0.0, 0.0
  for j in range(2):
    nsucess = 0  
    fig, ax = plt.subplots()
    data = DATA[j]
    dlen = len(data)
    x, y = np.zeros((2,dlen)), np.zeros((2,dlen))
    for k in range(dlen):
       elen = len(data[k])
       x[j][k] = data[k][1]
       y[j][k] = data[k][elen-1]
       if y[j][k] < 0.5: nsucess +=1 # the number of sucessful attack
    plt.scatter(x[j], y[j], label=label[j], alpha=0.5)
    cbar = plt.colorbar()
    cdiff = np.sum(x[j] - y[j])
    wdiff = np.sum(x[j]*(x[j] - y[j]))

    fig.subplots_adjust(bottom=0.2) 
    plt.xlabel('Confidence before attack' + '\ncdiff='+format(cdiff/len(data),'.4f') + ' wdiff=' + format(wdiff/len(data),'.4f') + ' sucess=' + format(nsucess/len(data), '.4f'))
    plt.ylabel('Confidence after attack')
    stitle = post_fn.split('-')
    stitle[6]= str(dlen)
    st = '-'.join(stitle) 
    plt.title(types[0] + st +  str(j) + '-' + TIME[j])
    plt.savefig(path+types[0]+ st + str(j)+'.png')
    plt.cla()
    plt.clf()
    plt.close('all')


# draw the scater of segmentation attacking samples, draw top-k sucess attacks
# from the pickle data
def draw_MRI_scatter(path='./results/', post_fn='EBL-1-5-0.5-1000-100-1000.pkl', save_topk=0):
  # get the Data 
  with open( path + post_fn, 'rb' ) as _f:
     DATA = pickle.load(_f)
  data = np.array( DATA ) 
  _pixels = list(set(data[:,1]))
  ## the columsn 5,6,7,8,2 is sucessful, cdiff, prior confidence/dice, original confidence/dice, ID
  _data = data[:,[5,6,7,8,2,9]].reshape(len(_pixels), len(data)//len(_pixels), 6)
  if save_topk>0:
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
 
  for j in range(len(_pixels)):
    nsucess = np.sum(_data[j][:,0])  
    cdiff = np.sum(_data[j][:,1])
    wdiff = np.sum( _data[j][:,1] * _data[j][:,2])
    fig, ax = plt.subplots() 
    plt.scatter(_data[j][:,2], _data[j][:,3], label='pixels='+str(_pixels[j]), alpha=0.5)
    cbar = plt.colorbar() 

    fig.subplots_adjust(bottom=0.2) 
    plt.xlabel('Dice before attack' + '\ncdiff='+format(cdiff/len(_data[j]),'.4f') + 
             ' wdiff=' + format(wdiff/len(_data[j]),'.4f') + ' sucess=' + format(nsucess/len(_data[j]), '.4f'))
    plt.ylabel('Dice after attack')
    _title = post_fn.replace('.pkl', '--'+str(len(_data[j]))+'-'+str(_pixels[j]))
    plt.title(_title)
    plt.savefig(path + _title+'.png')
    plt.cla()
    plt.clf()
    plt.close('all')
    
def draw_MRI_LP(path='results-1/', post_fn='EBL-1024-0.5-800-50-100--LP.pkl'): 
   f = open(path + post_fn, 'rb')
   data = np.array(pickle.load(f))
   f.close()
   sn = len(data)
   LP_data = np.zeros((3,sn))
   ave_pixels = 0
   for i in range(sn):
      if data[i,5]: ave_pixels += data[i][1]
      LP_data[0,i] = data[i][1]        # the number of pixels of the attack
      LP_data[1,i] = data[i][8]#[dtype] # the confidence after the attack
      LP_data[2,i] = data[i][7]#[dtype] # the confidence before the attack      
   cdiff, wdiff = np.sum(LP_data[2] - LP_data[1])/sn, np.sum((LP_data[2]-LP_data[1])/LP_data[0])/sn
   nsucess = np.sum(data[:,5]) / sn  
   labels = ['Dice before attack', 'Dice after attack','Number of attacking pixels']

   ax3d = plt.subplot(111,projection='3d')
   ax3d.scatter(LP_data[2], LP_data[1], LP_data[0])
   ax3d.set_xlabel(labels[0])
   ax3d.set_ylabel(labels[1])
   ax3d.set_zlabel(labels[2]) 
   plt.title('cdiff='+format(cdiff,'.4f') + ' wdiff=' + format(wdiff,'.4f') + ' success=' + format(nsucess, '.4f') + \
      '\nAverage attacked pixels=' + format(ave_pixels/np.sum(data[:,5]),'.4f') + ' time=3555m26.642s')
   plt.savefig(path + post_fn.replace('.pkl','-scatter.png'))
   plt.cla()
   plt.clf()
   plt.close('all')  
##
#[model.name, pixel_count, img_id, None, None, success, cdiff, prior_dice, attacked_dice, attack_result.x]
def draw_MRI_gray(_path='results/', post_fn='EBL-1024-0.5-800-50-100--LP.pkl', topk=5, least_pixels=False, color=plt.cm.gray, one_pixel=True):
 
    #get the model and data_loader
    model = utnet.UTNet(in_chan = 4, base_chan = 16, num_classes=2)
    device = torch.device(config.DEFAULT_DEVICE)
    model = model.to(device) 
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location = device))
    model.eval()
    torch.cuda.empty_cache()

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

    ## get the top_k original MRI image with the img_id
    f = open(_path + post_fn, 'rb')
    result = np.array( pickle.load(f) )
    result = result[result[:,5]==True]
    if not one_pixel:
       result = result[result[:,1]==5]
    else:
       result = result[result[:,1]==1]
    f.close()
    if least_pixels:
       index = np.argsort(result[:, 1])[0:topk]
    else:
       index = np.argsort(-result[:,7])[0:topk]
    #result = _result[np.lexsort(_result[_result[:,5]==True][:,::-7].T)][0:topk-1] 
    _post_name = ['original_', 'mask_', 'prediction_', 'perturbed_', 'perturbed_prediction_', 'perturbed_mask_']
    
    for k in range(min(topk, len(index))):
      # the original image: 4 channels, 2*2
      pixel_count, img_id, prior_dice, attacked_dice, xs = result[index[k],1], result[index[k],2], result[index[k],7], result[index[k],8], result[index[k],9].reshape(-1,6)
      batch_datas = data_loader[img_id]
      datas, mask = batch_datas
      img = torch.cat(datas, 1)

      fig, ax = plt.subplots(2,2,sharex=True, sharey=True)
      rec0 = Preprocess.ZeroOne_Normalization(img[0 , 0, : , :]) * 255    
      rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
      rec1 = Preprocess.ZeroOne_Normalization(img[0 , 1, : , :]) * 255    
      rec1 = rec1.int() ### for numpy.ndarray     astype(np.int64)
      rec2 = Preprocess.ZeroOne_Normalization(img[0 , 2, : , :]) * 255    
      rec2 = rec2.int() ### for numpy.ndarray     astype(np.int64)
      rec3 = Preprocess.ZeroOne_Normalization(img[0 , 3, : , :]) * 255    
      rec3 = rec3.int() ### for numpy.ndarray     astype(np.int64)            
      ax[0,0].imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
      ax[0,1].imshow(rec1.cpu().numpy(), origin='lower', cmap =color)
      ax[1,0].imshow(rec2.cpu().numpy(), origin='lower', cmap =color)
      ax[1,1].imshow(rec3.cpu().numpy(), origin='lower', cmap =color)      
      fig.suptitle(' Prior dice = ' + format(prior_dice,'.4f'))
      plt.savefig(_path + _post_name[0] + str(img_id) + '_'+ str(pixel_count) + '.png')
      plt.close('all')
      
      # the mask
      fig, ax = plt.subplots(1,1)
      rec0 = Preprocess.ZeroOne_Normalization(mask[0,0]) * 255    
      rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
      ax.imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
      plt.title(' The mask')
      plt.savefig(_path + _post_name[1] + str(img_id) + '_'+ str(pixel_count) + '.png')
      plt.close('all')

      # the prediction
      with torch.no_grad():
         pred = model(img)
      torch.cuda.empty_cache()
      fig, ax = plt.subplots(1,2)
      rec0 = Preprocess.ZeroOne_Normalization(pred[0 , 0, : , :]) * 255    
      rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
      rec1 = Preprocess.ZeroOne_Normalization(pred[0 , 1, : , :]) * 255    
      rec1 = rec1.int() ### for numpy.ndarray     astype(np.int64)
      ax[0].imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
      ax[1].imshow(rec1.cpu().numpy(), origin='lower', cmap =color)
      fig.suptitle('Prediction image')
      plt.savefig(_path + _post_name[2] + str(img_id) + '_'+ str(pixel_count) + '.png')
      plt.close('all')


      # the attacked image: 4 channels, 2*2
      perturb_img = perturb_MRI(xs.reshape(-1), img)
      fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
      rec0 = Preprocess.ZeroOne_Normalization(perturb_img[0 , 0, : , :]) * 255    
      rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
      rec1 = Preprocess.ZeroOne_Normalization(perturb_img[0 , 1, : , :]) * 255    
      rec1 = rec1.int() ### for numpy.ndarray     astype(np.int64)
      rec2 = Preprocess.ZeroOne_Normalization(perturb_img[0 , 2, : , :]) * 255    
      rec2 = rec2.int() ### for numpy.ndarray     astype(np.int64)
      rec3 = Preprocess.ZeroOne_Normalization(perturb_img[0 , 3, : , :]) * 255    
      rec3 = rec3.int() ### for numpy.ndarray     astype(np.int64)     
      if len(xs) <= 10:
         for j in range(len(xs)):  ## draw the attacked imgs with marked attack pixels positions 
            _py, _px = int(xs[j][0]), int(xs[j][1])
            ax[0,0].scatter(_px, _py, color='', edgecolor='w', marker='o', alpha=.5, s=150)
            ax[0,1].scatter(_px, _py, color='', edgecolor='w', marker='o', alpha=.5, s=150)
            ax[1,0].scatter(_px, _py, color='', edgecolor='w', marker='o', alpha=.5, s=150)
            ax[1,1].scatter(_px, _py, color='', edgecolor='w', marker='o', alpha=.5, s=150)
      ax[0,0].imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
      ax[0,1].imshow(rec1.cpu().numpy(), origin='lower', cmap =color)
      ax[1,0].imshow(rec2.cpu().numpy(), origin='lower', cmap =color)
      ax[1,1].imshow(rec3.cpu().numpy(), origin='lower', cmap =color)
      fig.suptitle('original dice: ' + format(prior_dice,'.4f') + ' dice after attack: ' + format(attacked_dice,'.4f'))
      plt.savefig(_path + _post_name[3] + str(img_id)  + '_' + str(pixel_count) +'.png')
      plt.close('all')

      # the prediction of attacked img
      with torch.no_grad():
         pred = model(perturb_img)
      torch.cuda.empty_cache()
      assert(abs(Dice.dice_values(pred, mask).cpu().numpy()-attacked_dice)<0.01)
      fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
      rec0 = Preprocess.ZeroOne_Normalization(pred[0 , 0, : , :]) * 255    
      rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
      rec1 = Preprocess.ZeroOne_Normalization(pred[0 , 1, : , :]) * 255    
      rec1 = rec1.int() ### for numpy.ndarray     astype(np.int64)
      ax[0].imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
      ax[1].imshow(rec1.cpu().numpy(), origin='lower', cmap =color)
      fig.suptitle('Prediction of perturbed image with dice: ' + format(attacked_dice,'.4f'))
      plt.savefig(_path + _post_name[4] + str(img_id) + '_'+ str(pixel_count) + '.png')
      plt.close('all')

      # the perturbed mask
      tpre = torch.argmax(pred, dim = 1, keepdim = True)
      tpre = torch.nn.functional.one_hot(tpre.long(), 2)
      tpre = torch.Tensor.permute(tpre, [0, 4, 2, 3, 1]) # to batch_size * target * size (length-width) * (background-foreground)
      tpre = tpre[:, 1, :, :, 0]

      fig, ax = plt.subplots(1,1)
      rec0 = Preprocess.ZeroOne_Normalization(tpre[0]) * 255    
      rec0 = rec0.int() ### for numpy.ndarray     astype(np.int64)
      ax.imshow(rec0.cpu().numpy(), origin='lower', cmap =color)
      plt.title(' The perturbed mask')
      plt.savefig(_path + _post_name[5] + str(img_id) + '_'+ str(pixel_count) + '.png')
      plt.close('all')

def draw_LP(path='./results/dr/', dtype=1, DE='EBL',post_fn='', sn=0):
   #path='./results/dr/'
   f = open(path + DE + post_fn + str(dtype) + '.pkl', 'rb')
   data = pickle.load(f)
   f.close()
   sn = len(data)
   LP_data = np.zeros((3,sn))
   for i in range(sn):
      LP_data[0,i] = data[i][1]        # the number of pixels of the attack
      LP_data[1,i] = data[i][8] #[dtype] # the confidence after the attack
      LP_data[2,i] = data[i][7] #[dtype] # the confidence before the attack
   cdiff, wdiff = np.sum(LP_data[2] - LP_data[1])/sn, np.sum((LP_data[2]-LP_data[1])/LP_data[0])/sn
   ylabels = ['The least number of pixels', 'The confiderence after attack']
   for i in range(2):
      fig, ax = plt.subplots()
      plt.scatter(LP_data[2], LP_data[i], alpha=0.5)
      plt.colorbar()
      plt.xlabel('Confidence before attack' + '\ncdiff='+format(cdiff,'.4f') + ' wdiff=' + format(wdiff,'.4f') )
      plt.ylabel(ylabels[i])
      fig.subplots_adjust(bottom=0.2)
      plt.title(DE + post_fn + str(dtype) + '-LP-' + format(np.sum(LP_data[0])/sn,'.2f'))
      plt.savefig(path + DE + post_fn + str(dtype) + '-LP-' + str(i) + '.png')
      plt.cla()
      plt.clf()
      plt.close('all')

   # 3D
   fig, ax = plt.subplots(subplot_kw=dict(projection='3d')) 
   cmap = cm.viridis
   ax.scatter(LP_data[2], LP_data[1], LP_data[0], c='b', cmap=cmap, alpha=0.4, linewidth=0)  
   #fig.colorbar()
   ax.set_zlabel('Number of attacked pixels') # 坐标轴
   ax.set_ylabel('Confidence after  attack')
   ax.set_xlabel('Confidence before attack') 
   plt.title('cdiff='+format(cdiff,'.4f') + ' wdiff=' + format(wdiff,'.4f') +  ' LP=' + format(npixels/sn,'.2f') + ' success=' + format(nsucess/len(data), '.4f'))
   plt.savefig(path + DE + post_fn + str(dtype) + '-LP-3D.png')
   plt.close()

if __name__ == '__main__': 
    
   draw_MRI_scatter(post_fn='EBL-1-5-0.5-1000-100-1000.pkl')
