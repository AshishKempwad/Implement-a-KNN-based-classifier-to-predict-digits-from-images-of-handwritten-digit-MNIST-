#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# In[2]:


class KNNClassifier:
    def _init_(self):
        self.k=0
        self.train_data=[]
        
        
        
    def train(self,path):  
        dataset=pd.read_csv(path,header=None)
        self.train_data=dataset
        x=dataset.iloc[:,:1]
        y=dataset.iloc[:,1:]
        pixel=pd.DataFrame(y).to_numpy()
        label=pd.DataFrame(x).to_numpy()
        self.k=5
        
    
    def predict(self,path):
        test_data=pd.read_csv(path,header=None) 
        z=test_data
        test_pixel=pd.DataFrame(z).to_numpy()
        dataset=self.train_data
        x=dataset.iloc[:,:1]
        y=dataset.iloc[:,1:]

#         data=pd.DataFrame(X_train).to_numpy()
#         y=data[:,1:]
#        x=X_train[:,:1]
        pixel=pd.DataFrame(y).to_numpy()
        label=pd.DataFrame(x).to_numpy()
        
        list2=[]
        for i in range(len(test_pixel)):
            list1=[]
            neighbors = []
            m=test_pixel[i]
            for j in range(len(pixel)):
                l=pixel[j]
                q=label[j]
                distance = np.linalg.norm(l-m)
                list1.append((q,distance))

            list1.sort(key=lambda ele:ele[1])
            kv=self.k
            for p in range(kv):
                neighbors.append(list1[p][0])

            output_values = [row[-1] for row in neighbors]
            prediction = max(set(output_values), key=output_values.count) 
            list2.append(prediction)
        return list2
        
        
       




