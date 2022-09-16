import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np

def read_st(fn, n):
# fn: file name
# n: number of samples
# read the data from the file fn and return the data

  # get the number sample from pickle file
  _f = open(fn.replace('txt','pkl'), 'rb')
  _d = pickle.load(_f)
  sn = len(_d)
  _f.close()

  Data = [] # for all samples
  f = open(fn,"r")

  # skip the two lines:
  # Evaluating model_1
  # Starting attack
  line = f.readline()
  line = f.readline()

  line = f.readline().split()
  for i in range(sn):
     data = [] # for one sample
     data.append( float(line[5] ) )
     data.append( float( f.readline().split()[3] ))
     line = f.readline().split()
     while(len(line) == 5):
        data.append( float(line[4]) )
        line = f.readline().split()
     # skip the line 
     # The minimal number of successful attacking pixels is 
     if len(line) != 8:  line = f.readline().split()
     Data.append(data)
  # get the time
  line = f.readline()
  while(line.rfind('user') == -1): 
     line = f.readline()
  user = line.split()[1]
  sys  = f.readline().split()[1]
  m1, s1 = user.split('m')[0], user.split('m')[1].split('s')[0]
  m2, s2 = sys.split('m')[0], user.split('m')[1].split('s')[0]
  Time = str(int(m1)+int(m2)) + 'm' + format(float(s1)+float(s2),'.2f') + 's'
  #line.split()[1]
  f.close()
  return Data, Time


def draw_1000s(post_fn='-0-1-0.01-500-1000-10-1'):
  # get the Data
  types = ['DE', 'SHA', 'EBL']
  path='./results/dr/'
  #post_fn = '-0-1-0.01-500-1000-10-1'
  DATA = []
  TIME = []
  for i in range(3):
     _data,_time = read_st(path + types[i] + post_fn + '.txt',10)
     DATA.append(_data)
     TIME.append(_time)

  # print the original confidence
  for i in range(len(DATA[0])):
     print('ID=',i, '  Confidence=', DATA[0][i][1]) 
  # draw DE, SHA, EBL
  for i in range(3):
     data = np.array(DATA[i])  
     fig, ax = plt.subplots() 
     cdiff,wdiff = 0.0, 0.0
     for j in range(len(data)):
        plt.plot(data[j][2:], label=str(j)+': '+format(data[j][1],'.4f')) 
        cdiff += (data[j][1]-data[j][len(data[j])-1])
        wdiff += data[j][1]*(data[j][1]-data[j][len(data[j])-1])

     fig.subplots_adjust(right=0.7, bottom=0.2)
     plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0) #loc='lower right',mode='expand')
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
     for j in range(3):
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


def draw_LP(path='./results/dr/', dtype=1, DE='EBL',post_fn='', sn=0):
   #path='./results/dr/'
   f = open(path + DE + post_fn + str(dtype) + '.pkl', 'rb')
   data = pickle.load(f)
   f.close()
   sn = len(data)
   LP_data = np.zeros((3,sn))
   nsucess, npixels = 0, 0
   for i in range(sn):
      LP_data[0,i] = data[i][1]        # the number of pixels of the attack
      LP_data[1,i] = data[i][8][dtype] # the confidence after the attack
      LP_data[2,i] = data[i][7][dtype] # the confidence before the attack
      if data[i][8][dtype] < 0.5: 
         nsucess += 1
         npixels += data[i][1]
 
   cdiff, wdiff = np.sum(LP_data[2] - LP_data[1])/sn, np.sum((LP_data[2]-LP_data[1])/LP_data[0])/sn
   # 
   ylabels = ['The least number of pixels', 'The confiderence after attack']
   for i in range(2):
      fig, ax = plt.subplots()
      plt.scatter(LP_data[2], LP_data[i], alpha=0.5)
      plt.colorbar()
      plt.xlabel('Confidence before attack' + '\ncdiff='+format(cdiff,'.4f') + ' wdiff=' + format(wdiff,'.4f') + ' success=' + format(nsucess/len(data), '.4f') )
      plt.ylabel(ylabels[i])
      fig.subplots_adjust(bottom=0.2)
      plt.title(DE + post_fn + str(dtype) + '-LP-' + format(npixels/nsucess,'.2f'))
      plt.savefig(path + DE + post_fn + str(dtype) + '-LP-' + str(i) + '.png')
      plt.cla()
      plt.clf()
      plt.close('all')
   ## draw 3d scatter plots

   #ax = plt.subplot(projection='3d')     
   # for xx, yy, zz in zip(LP_data[2], LP_data[1], LP_data[0]):
   #  color = np.random.random(3)   # 随机颜色元祖
   #  ax.bar3d(
   #      xx,            # 每个柱的x坐标
   #      yy,            # 每个柱的y坐标
   #      0,             # 每个柱的起始坐标
   #      dx=1,          # x方向的宽度
   #      dy=1,          # y方向的厚度
   #      dz=zz)#,         # z方向的高度
   #      #color=color)   #每个柱的颜色
   fig, ax = plt.subplots(subplot_kw=dict(projection='3d')) 
   cmap = cm.viridis
   ax.scatter(LP_data[2], LP_data[1], LP_data[0], c='b', cmap=cmap, alpha=0.4, linewidth=0)  
   #fig.colorbar()
   ax.set_zlabel('Number of attacking pixels') # 坐标轴
   ax.set_ylabel('Confidence after attack')
   ax.set_xlabel('Confidence before attack') 
   plt.title('cdiff='+format(cdiff,'.4f') + ' wdiff=' + format(wdiff,'.4f') +  ' LP=' + format(npixels/nsucess,'.2f') + ' success=' + format(nsucess/len(data), '.4f'))
   plt.savefig(path + DE + post_fn + str(dtype) + '-LP-3D.png')
   plt.close()


def draw_RA(source='dr',type=0, RAn=10, RA=True):
        for i in range(10): #min(len(_ids), args.RAn)):
           _path = './results/' + source + '/'
           _file_name = _path + 'RA-' +str(i) + '-' + str( int(_ids[i][0]) ) + '-' + str(type) + '.png'

           # the original image
           fix,ax = plt.subplots(1)
           ax.imshow(_test[i], origin='lower')
           plt.title(' Prior conf=' + format(RA_results[i][7][type],'.4f'))
           plt.xlabel('ID='+str(int(_ids[i][0])))
           plt.savefig(_path + 'RA-ori-' + str(i) + '-' + str( int(_ids[i][0]) ) + '-' + str(type) + '.png')
           plt.close('all')

           # the attacked image
           fig,ax = plt.subplots(1)
           ax.set_aspect('equal')
           _cdiff = 0.0
           
           # get the attacking pixels 
           ## draw circles of repeated attacks 
           for j in range(RAn):
              _px, _py = int(RA_results[j * RAn + i][9][0]), int(RA_results[j * RAn + i][9][1])
              _cdiff += RA_results[j * RAn + i][6] 
              plt.scatter(_px, _py, color='', edgecolor='b', marker='o', alpha=.5, s=150)
              _test[i][_py,_px] = RA_results[j * RAn + i][9][2], RA_results[j * RAn + i][9][3], RA_results[j * RAn + i][9][4]  

           _title = ' cdiff=' + format(_cdiff/RAn,'.4f')
           plt.title(_title)
           plt.xlabel('ID='+str(int(_ids[i][0])))
           ax.imshow(_test[i], origin='lower')
           plt.savefig(_file_name)
           plt.close('all') 

if __name__ == '__main__':  
   draw_1000s('-0-1-0.01-1000-1000-10-1')
   draw_1000s('-0-1-0.01-1000-1000-10-0') 