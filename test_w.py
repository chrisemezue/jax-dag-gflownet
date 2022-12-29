import numpy as np
import os
import pandas as pd





def read_adj_mat(path):
    adj_mat = np.load(path)
    return adj_mat


# https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}


# Get X
# get the corresponding posterior for it.

BASE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20/bcdnets/5'
indices = [k for k,v in node_to_index.items()]

data = pd.read_csv(os.path.join(BASE_FOLDER,'data.csv'))[indices].to_numpy()
adj_matrix_ = read_adj_mat(os.path.join(BASE_FOLDER,'posterior.npy'))
adj_matrix = None
for i in range(adj_matrix_.shape[2]):

    all_zeros = not np.any(adj_matrix_[i])
    if all_zeros == False:
        adj_matrix = adj_matrix_[i]
        continue

import pdb;pdb.set_trace()

# X = XWoA | b = Ax. Let WoA = U, then we have X = XU
u = np.linalg.lstsq(a=data,b=data)
import pdb;pdb.set_trace()