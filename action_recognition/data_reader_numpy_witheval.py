import sys
import h5py
import numpy as np
import time
import random
#import glob
#import skimage.transform
#from skimage import color
import pickle
#import theano
#import cv2
from multiprocessing import Pool
from threading import Thread
import os.path
#RGB_frames = '/home/sl669/caffe/colordataset/ImageNET/ILSVRC2015/Data/CLS-LOC/val/'#'/home/sl669/caffe/ucf101/framearrays/'#

from __main__ import train_datasets
#train_datasets='train_ntus'
datasets=train_datasets
dataname=datasets+'.npy'
labelname=datasets+'_label.npy'
lenname=datasets+'_len.npy'
data_handle=np.load(dataname)
label_handle=np.load(labelname)
len_handle=np.load(lenname)
num_videos = len(data_handle)  
train_no=int(num_videos*0.95)
test_no=num_videos-train_no

shufflevideolist=np.arange(num_videos)
np.random.shuffle(shufflevideolist)

shufflevideolist_train=shufflevideolist[:train_no]
shufflevideolist_test=shufflevideolist[train_no:]

print ('Dataset train size, test size', train_no,test_no)


def rotate( input,s,b):
  shape=input.shape
  input=input.reshape((-1,3))
  XT=input[:,0]
  YT=input[:,1]
  ZT=input[:,2]
  s=s/180.0*np.pi
  b=b/180.0*np.pi
  RX = XT*np.cos(b) - ZT*np.sin(b) + ZT*np.sin(b)*np.cos(s) + YT*np.sin(b)*np.sin(s) - ZT*np.sin(b)*(np.cos(s) - 1);
  RY = YT*np.cos(s);
  RZ = ZT*np.cos(b)*np.cos(s) - ZT*(np.cos(b) - 1) - XT*np.sin(b) + YT*np.cos(b)*np.sin(s) - ZT*np.cos(b)*(np.cos(s) - 1);
  RX=RX.reshape((-1,1))
  RY=RY.reshape((-1,1))
  RZ=RZ.reshape((-1,1))
  output=np.concatenate([RX,RY,RZ],axis=1)
  output=output.reshape(shape)
  #print(shape,output.shape,input.shape)
  return output 

class batch_thread_train():
  def __init__(self, result, batch_size_,seq_len,use_rotation=False):
    self.result = result
    self.batch_size_=batch_size_
    self.seq_len=seq_len
    self.idx=0    
    self.use_rotation=use_rotation
  
  def __call__(self):###Be careful.  The appended data may change like pointer.
    templabel=[] 
    batch_data=[]
    for j in range(self.batch_size_):
      self.idx +=1
      if self.idx == train_no:
        self.idx =0
        np.random.shuffle(shufflevideolist_train)
      shufflevideoindex=shufflevideolist_train[self.idx]
      
      
      label=label_handle[shufflevideoindex]     
      templabel.append(np.int32(label))  
      dataset=data_handle[shufflevideoindex]
      len_data=len_handle[shufflevideoindex]   
      
      sample=np.zeros(tuple((self.seq_len,)+data_handle[shufflevideoindex].shape[1:]))
      lenperseg=len_data//self.seq_len
      if lenperseg==1 and len_data>self.seq_len:
        startid=np.random.randint(len_data-self.seq_len)
        sample=dataset[startid:startid+self.seq_len]
        #print('wrong data length first')
      elif len_data<=self.seq_len:
        startid=np.random.randint(max(self.seq_len-len_data,int(0.25*self.seq_len)))
        endid=min(self.seq_len,startid+len_data)
        datasid=0
        dataeid=len_data
        if startid+len_data>self.seq_len:
          datasid=np.random.randint(startid+len_data-self.seq_len)
          dataeid=datasid+self.seq_len-startid
        sample[startid:endid]=dataset[datasid:dataeid]
      else:      
        for framei in range(self.seq_len):        
          if framei==self.seq_len-1:
            index=lenperseg*framei + np.random.randint(len_data-lenperseg*(self.seq_len-1))
          else:
            index=lenperseg*framei + np.random.randint(lenperseg)    
          sample[framei]=dataset[index]
          
      #print(sample)
      if self.use_rotation:
        if np.random.randint(2):
          s=np.random.randint(2)*45#random(1)*45
          b=np.random.randint(2)*45#random(1)*45
          #print(sample.shape)
          sample=rotate(sample,s,b)
        #print (index,lenperseg)  
#       rframei=np.random.randint(len_data)  
#       tmean=(dataset[rframei,0,:]+dataset[rframei,12,:]+dataset[rframei,16,:])/3
#       sample=sample-tmean  
      batch_data.append(sample) ###Be careful. It has to be different. Otherwise, the appended data will change as well.
      #print(batch_data)       
      
    self.result['data']=np.asarray(batch_data,dtype=np.float32)
    self.result['label']= np.asarray(templabel,dtype=np.int32)   

class DataHandler_train(object):

  def __init__(self, batch_size, seq_len, use_rotation=False):#datasets,
    self.batch_size_ = batch_size		
    #self.datasets = datasets    
    random.seed(10)  
    
    self.thread_result = {}
    self.thread = None

    self.batch_advancer =batch_thread_train(self.thread_result,self.batch_size_,seq_len,use_rotation)
    
    self.dispatch_worker()
    self.join_worker()


  def GetBatch(self):
    #self.batch_data_  = np.zeros((self.batch_size_, 3, self.seq_length_, 112, 112), dtype=np.float32)
    if self.thread is not None:
      self.join_worker() 

    self.batch_data_=self.thread_result['data']
    self.batch_label_= self.thread_result['label']
        
    self.dispatch_worker()
    return self.batch_data_, self.batch_label_

  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None
    
  def GetDatasetSize(self):
    return train_no









class batch_thread_eval():
  def __init__(self, result, batch_size_,seq_len):
    self.result = result
    self.batch_size_=batch_size_
    self.seq_len=seq_len
    self.idx=0    
  
  def __call__(self):###Be careful.  The appended data may change like pointer.
    templabel=[] 
    batch_data=[]
    for j in range(self.batch_size_):
      self.idx +=1
      if self.idx == test_no:
        self.idx =0
        np.random.shuffle(shufflevideolist_test)
      shufflevideoindex=shufflevideolist_test[self.idx]
      
      
      label=label_handle[shufflevideoindex]     
      templabel.append(np.int32(label))  
      dataset=data_handle[shufflevideoindex]
      len_data=len_handle[shufflevideoindex]   
      
      sample=np.zeros(tuple((self.seq_len,)+data_handle[shufflevideoindex].shape[1:]))
      lenperseg=len_data//self.seq_len
      if lenperseg==1 and len_data>self.seq_len:
        startid=np.random.randint(len_data-self.seq_len)
        sample=dataset[startid:startid+self.seq_len]
      elif len_data<=self.seq_len:
        startid=np.random.randint(max(self.seq_len-len_data,int(0.25*self.seq_len)))
        endid=min(self.seq_len,startid+len_data)
        datasid=0
        dataeid=len_data
        if startid+len_data>self.seq_len:
          datasid=np.random.randint(startid+len_data-self.seq_len)
          dataeid=datasid+self.seq_len-startid
        sample[startid:endid]=dataset[datasid:dataeid]
      else:      
        for framei in range(self.seq_len):        
          if framei==self.seq_len-1:
            index=lenperseg*framei + np.random.randint(len_data-lenperseg*(self.seq_len-1))
          else:
            index=lenperseg*framei + np.random.randint(lenperseg)    
          sample[framei]=dataset[index]
        #print (index,lenperseg)  
        
      batch_data.append(sample) ###Be careful. It has to be different. Otherwise, the appended data will change as well.
      #print(batch_data)       
      
    self.result['data']=np.asarray(batch_data,dtype=np.float32)
    self.result['label']= np.asarray(templabel,dtype=np.int32)   

class DataHandler_eval(object):

  def __init__(self, batch_size, seq_len):#, datasets
    self.batch_size_ = batch_size    
    #self.datasets = datasets    
    random.seed(10)  
    
    self.thread_result = {}
    self.thread = None

    self.batch_advancer =batch_thread_eval(self.thread_result,self.batch_size_,seq_len)
    
    self.dispatch_worker()
    self.join_worker()


  def GetBatch(self):
    #self.batch_data_  = np.zeros((self.batch_size_, 3, self.seq_length_, 112, 112), dtype=np.float32)
    if self.thread is not None:
      self.join_worker() 

    self.batch_data_=self.thread_result['data']
    self.batch_label_= self.thread_result['label']
        
    self.dispatch_worker()
    return self.batch_data_, self.batch_label_

  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None
    
  def GetDatasetSize(self):
    return test_no



def main():
  dh = DataHandler_train(1, 30,True)#'test_ntus.h5')#'test_ntus_allwitherror.h5')#
  print (dh.GetDatasetSize())
  dh_eval = DataHandler_eval(10, 30)#'test_ntus.h5')#'test_ntus_allwitherror.h5')#
  print (dh_eval.GetDatasetSize())
 
  x,y = dh.GetBatch()
#   print (x.shape)
#   print (y[0:3],x[0,0,0],x[1,0,0],x[0,1,0])
#   x,y = dh_eval.GetBatch()
#   #print (x[0,0],y)  
#   print (y,x[0,0,0])
#   x,y = dh.GetBatch()
#   #print (x[0,0],y)
#   print (y,x[0,0,0])
  x,y = dh.GetBatch()
  #print (x[0,0],y)    
  #print (y,x[0,0,0])
#   exit()

if __name__ == '__main__':
  main()

