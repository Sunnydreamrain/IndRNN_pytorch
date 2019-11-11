import sys
import h5py
import numpy as np
import time
import random
import pickle
from multiprocessing import Pool
from threading import Thread
import os.path

line_set=np.array([
[4,3],[3,21],[21,2],[2,1],[21,9],[9,10],[10,11],[11,12],[12,24],[12,25],[21,5],[5,6],[6,7],
[7,8],[8,22],[8,23],[1,17],[17,18],[18,19],[19,20],[1,13],[13,14],[14,15],[15,16],[19,17],[12,9],
[15,13],[5,8],[4,2],[19,15],[19,4],[19,12],[19,5],[15,4],[15,12],[15,5],[4,12],[4,5],[12,5],
],dtype=int)
def calculate_JL_d(inputs,max_body=2,num_joint=25):
    num_line=39 #np.size(line_set,0) 

    batch_size, seq_len, joints_no,_=inputs.shape
    output=np.zeros((batch_size, seq_len,  897*2),dtype=np.float16)
    count=0
    
    for line in line_set:
        j1_idx=line[0]-1
        j2_idx=line[1]-1

        j1=inputs[:,:,j1_idx,:]#inputs[f,body*num_joint+j1_idx]
        j2=inputs[:,:,j2_idx,:] #inputs[f,body*num_joint+j2_idx]
        len_j1j2=np.linalg.norm(j1-j2,ord=2,axis=-1)
        
        j1_s2=inputs[:,:,j1_idx+25,:]#inputs[f,body*num_joint+j1_idx]
        j2_s2=inputs[:,:,j2_idx+25,:] #inputs[f,body*num_joint+j2_idx]
        len_j1j2_s2=np.linalg.norm(j1_s2-j2_s2,ord=2,axis=-1)


        for j0_idx in range(0,num_joint):
            if j0_idx==j1_idx or j0_idx==j2_idx:
                continue
            j0=inputs[:,:,j0_idx,:]#inputs[f,j0_idx]
            j0_s2=inputs[:,:,j0_idx+25,:]#inputs[f,j0_idx+25]

            len_j0j1=np.linalg.norm(j0-j1,ord=2,axis=-1)
            len_j0j2=np.linalg.norm(j0-j2,ord=2,axis=-1)
            len_p=(len_j1j2+len_j0j1+len_j0j2)/2

            len_j0j1_s2=np.linalg.norm(j0_s2-j1_s2,ord=2,axis=-1)
            len_j0j2_s2=np.linalg.norm(j0_s2-j2_s2,ord=2,axis=-1)
            len_p_s2=(len_j1j2_s2+len_j0j1_s2+len_j0j2_s2)/2

            s=np.sqrt(len_p*(len_p-len_j1j2)*(len_p-len_j0j1)*(len_p-len_j0j2))
            s_s2=np.sqrt(len_p_s2*(len_p_s2-len_j1j2_s2)*(len_p_s2-len_j0j1_s2)*(len_p_s2-len_j0j2_s2))
            output[:,:,count]=np.float16(2*s/(len_j1j2+1e-5))
            output[:,:,count+1]=np.float16(2*s_s2/(len_j1j2_s2+1e-5))
            count=count+2

    if count!=897*2:
        print('geo error')
        assert 2==3
    output = np.nan_to_num(output)
    return output

calculate_geo_aug=calculate_JL_d



from __main__ import train_datasets,test_dataset,geo_aug,data_randtime_aug
#train_datasets='train_ntus'
datasets=train_datasets
dataname=datasets+'.npy'
labelname=datasets+'_label.npy'
lenname=datasets+'_len.npy'
train_data_handle=np.load(dataname)
train_label_handle=np.load(labelname)
train_len_handle=np.load(lenname)
num_videos = len(train_data_handle)  
train_no=int(num_videos*0.95)
eval_no=num_videos-train_no

shufflevideolist=np.arange(num_videos)
np.random.shuffle(shufflevideolist)

shufflevideolist_train=shufflevideolist[:train_no]
shufflevideolist_eval=shufflevideolist[train_no:]

if geo_aug:
  train_data_handle=np.asarray(train_data_handle,dtype=np.float16)
  if not os.path.exists(datasets+'_geo_data.npy'):
    geo_data=calculate_geo_aug(train_data_handle)
    np.save(datasets+'_geo_data',geo_data)
    print('train geo data saved')
  else:
    geo_data=np.load(datasets+'_geo_data.npy')
    print('train geo data loaded')
  batch_size, seq_len, _=geo_data.shape
  geo_data=np.reshape(geo_data,(batch_size, seq_len, -1,3))
  train_data_handle=np.concatenate([geo_data,train_data_handle],axis=2)

datasets=test_dataset
dataname=datasets+'.npy'
labelname=datasets+'_label.npy'
lenname=datasets+'_len.npy'
test_data_handle=np.load(dataname)
test_label_handle=np.load(labelname)
test_len_handle=np.load(lenname) 

test_no = len(test_data_handle)    
shufflevideolist_test=np.arange(test_no)
np.random.shuffle(shufflevideolist_test)

print ('Dataset train size, eval size, test size', train_no,eval_no,test_no)


if geo_aug:
  test_data_handle=np.asarray(test_data_handle,dtype=np.float16)
  if not os.path.exists(datasets+'_geo_data.npy'):
    geo_data=calculate_geo_aug(test_data_handle)
    np.save(datasets+'_geo_data',geo_data)
    print('test geo data saved')
  else:
    geo_data=np.load(datasets+'_geo_data.npy')
    print('test geo data loaded')
  batch_size, seq_len, _=geo_data.shape
  geo_data=np.reshape(geo_data,(batch_size, seq_len, -1,3))
  test_data_handle=np.concatenate([geo_data,test_data_handle],axis=2)

class batch_thread():
  def __init__(self, result, batch_size_,seq_len,train_or_eval):
    self.result = result
    self.batch_size_=batch_size_
    self.seq_len_ori=seq_len
    self.idx=0    
    if train_or_eval=='train':
      self.data_list=shufflevideolist_train
      self.data_handle=train_data_handle
      self.label_handle=train_label_handle
      self.len_handle=train_len_handle
    elif  train_or_eval=='eval':
      self.data_list=shufflevideolist_eval
      self.data_handle=train_data_handle
      self.label_handle=train_label_handle
      self.len_handle=train_len_handle
    elif  train_or_eval=='test':
      self.data_list=shufflevideolist_test
      self.data_handle=test_data_handle
      self.label_handle=test_label_handle
      self.len_handle=test_len_handle
  
  def __call__(self):###Be careful.  The appended data may change like pointer.
    templabel=[] 
    batch_data=[]
    tempindex=[] 

    if data_randtime_aug:
      self.seq_len= int(np.random.normal(self.seq_len_ori, 5))#5,50
      if self.seq_len<5 or self.seq_len>40:
        self.seq_len = int(np.random.normal(self.seq_len_ori, 5))
      if self.seq_len<5 or self.seq_len>40:
        self.seq_len = self.seq_len_ori
    else:
      self.seq_len = self.seq_len_ori

    for j in range(self.batch_size_):
      self.idx +=1
      if self.idx == len(self.data_list):
        self.idx =0
        np.random.shuffle(self.data_list)
      shufflevideoindex=self.data_list[self.idx]
      
      
      label=self.label_handle[shufflevideoindex]     
      templabel.append(np.int32(label))  
      tempindex.append(np.int32(shufflevideoindex)) 
      dataset=self.data_handle[shufflevideoindex]
      len_data=self.len_handle[shufflevideoindex]   
      
      sample=np.zeros(tuple((self.seq_len,)+dataset.shape[1:]))
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
          
      batch_data.append(sample) ###Be careful. It has to be different. Otherwise, the appended data will change as well.
      #print(batch_data)       
      
    self.result['data']=np.asarray(batch_data,dtype=np.float32)
    self.result['label']= np.asarray(templabel,dtype=np.int32)    
    self.result['index']= np.asarray(tempindex,dtype=np.int32)  

class DataHandler(object):

  def __init__(self, batch_size, seq_len, train_or_eval, use_rotation=False):#datasets,
    self.batch_size_ = batch_size		
    #self.datasets = datasets    
    random.seed(10)  
    
    self.thread_result = {}
    self.thread = None
    self.train_or_eval=train_or_eval

    self.batch_advancer =batch_thread(self.thread_result,self.batch_size_,seq_len,train_or_eval)
    
    self.dispatch_worker()
    self.join_worker()


  def GetBatch(self):
    #self.batch_data_  = np.zeros((self.batch_size_, 3, self.seq_length_, 112, 112), dtype=np.float32)
    if self.thread is not None:
      self.join_worker() 

    self.batch_data_=self.thread_result['data']
    self.batch_label_= self.thread_result['label']
    self.batch_index_= self.thread_result['index']
        
    self.dispatch_worker()
    if self.train_or_eval=='test':
      return self.batch_data_, self.batch_label_,self.batch_index_
    else:
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
    if self.train_or_eval=='train':
      return len(shufflevideolist_train)
    elif  self.train_or_eval=='eval':
      return len(shufflevideolist_eval)
    elif  self.train_or_eval=='test':
      return len(shufflevideolist_test)







def main():
  dh = DataHandler(1, 30,True)#'test_ntus.h5')#'test_ntus_allwitherror.h5')#
  print (dh.GetDatasetSize())
 
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

