import numpy as np
import re
from numpy.core.defchararray import startswith
from numpy.core.numeric import NaN
import pandas as pd
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
path = 'project/dna_count.txt'
fr = open(path)
lines = fr.readlines()
dataset = []
for line in lines:
    line = line.strip()
    line = re.split(r'\t', line)
    line = [np.NaN if i.endswith("nan") else i for i in line]
    line = [i if i=="ds" or i=="ss"  else float(i) for i in line]
    dataset.append(line)

dataset = [line[0:32] for line in dataset]
rbs_names=[]
for i in range(28):
    rbs_names.append("RBS"+str(i+1))
dna_header=["length","skew","gc_content"]
dna_header=dna_header+rbs_names+["type"]
cols_used = dna_header[2:31]
dataset_array = np.array(dataset)
dataset_pd = pd.DataFrame(dataset_array, columns=dna_header)
dataset_pd = dataset_pd.dropna(how='all')
dna_data = dataset_pd.get(cols_used)
for item in cols_used:
    dna_data[item]=dna_data[item].dropna()
    dna_data[item]=pd.to_numeric(dna_data[item])
labels = dataset_pd.get('type')
dna_data = np.array(dna_data)

dna_data = (dna_data - dna_data.min(0))/dna_data.ptp(0)

def pca(x,dim = 2):
    with tf.name_scope("PCA"):
        
        m,n= tf.to_float(x.get_shape()[0]),tf.to_int32(x.get_shape()[1])
        print(n)
        assert not tf.assert_less(dim,n)
        mean = tf.reduce_mean(x,axis=1)
        print(mean)
        x_new = x - tf.reshape(mean,(-1,1))
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(m - 1) 
        e,v = tf.linalg.eigh(cov,name="eigh")
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1]
        v_new = tf.gather(v,indices=e_index_sort)
        pca = tf.matmul(x_new,v_new,transpose_b=True)
    return pca


pca_data = tf.constant(np.reshape(dna_data,(-1,29)),dtype=tf.float32)
# pca_data
pca_data=pca(pca_data)
print(pca_data[:,0])
