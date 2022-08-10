import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from dowhy import CausalModel
import pickle as pl
from typing import Optional

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 

baselines_ = []
rmses_ = []
seeds_ = []


BASELINE_FOLDER = '/home/mila/c/chris.emezue/scratch/baselines/'



def get_causal_estimate(graph,df):
    model = CausalModel(data=df, treatment=['K'],outcome='M',graph=nx.DiGraph(graph),use_graph_as_is=True)
    # II. Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(identified_estimand,
            target_units='atc',
            control_value=0,
            treatment_value=1,
            method_name="backdoor.linear_regression",
            test_significance=True,confidence_intervals=True)

    causal_estimate = causal_estimate_reg.value
    #print("Causal Estimate is " + str(causal_estimate_reg.value))
    return causal_estimate

def get_estimate_from_posterior(each_posterior):
    graph_sample = nx.from_numpy_array(each_posterior,create_using=nx.DiGraph)
    if nx.is_directed_acyclic_graph(graph_sample): # Check it is acyclic DAG
        graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
        estimate = get_causal_estimate(graph_sample_relabeled,df)
        return estimate
    else:
        return None    


def calculate_rmse(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the root mean squared error (RMSE) between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.sqrt(np.mean(np.square(np.subtract(a, b)), axis=axis))


baseline_to_use = sys.argv[1]
SEED_TO_USE =5
for baseline in [baseline_to_use]:
    for seed in [SEED_TO_USE]:
        BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

        with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
            graph = pl.load(fl)

        df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

        true_estimate = get_causal_estimate(graph,df)

        # Get posterior
        count=0
        posterior_file_path = os.path.join(BASE_PATH,'posterior.npy')
        if not os.path.isfile(posterior_file_path):
            continue
        posterior = np.load(posterior_file_path)
        st_time = time.time()
        causal_estimates = np.array([get_estimate_from_posterior(posterior[i,:,:]) for i in range(posterior.shape[0])])
        with open(f'/home/mila/c/chris.emezue/scratch/ate_estimates/{baseline_to_use}_{seed}_ate_estimates.npy', 'wb') as fl:
            np.save(fl,causal_estimates)
        #causal_estimates = np.full(posterior.shape[0], fill_value=1) 
        print(f'Time taken for one posterior of size {posterior.shape[0]} was {time.time()-st_time}')
        true_causal_estimates = np.full(causal_estimates.shape, fill_value=true_estimate)
        with open(f'/home/mila/c/chris.emezue/scratch/ate_estimates/true_{baseline_to_use}_{seed}_ate_estimates.npy', 'wb') as fl:
            np.save(fl,true_causal_estimates)
        rmse_value = calculate_rmse(causal_estimates,true_causal_estimates)
        
        baselines_.append(baseline)
        rmses_.append(rmse_value)
        seeds_.append(seed)

        
df = pd.DataFrame({'baselines':baselines_,'rmse':rmses_,'seeds':seeds_})
df.to_csv(f'ate_estimates/{baseline_to_use}_ate_estimates.csv',index=False)
print('ALL DONE')



'''
(Pdb) df.max(axis=0)
Unnamed: 0    99.000000
A              0.548883
B              0.331622
C              0.286164
D              0.933619
E              0.192666
F              0.239168
G              0.238359
H              0.421542
I              1.018528
J              0.314951
K              0.315432
L              0.247244
M              1.427548
N              0.354442
O              0.288996
P              0.248688
Q              0.217793
R              0.442156
S              0.477876
T              0.574680
________________________________
Min

(Pdb) df.min(axis=0)
Unnamed: 0    0.000000
A            -0.487034
B            -0.340195
C            -0.228957
D            -1.053950
E            -0.389942
F            -0.242183
G            -0.251634
H            -0.482522
I            -1.022796
J            -0.304008
K            -0.463249
L            -0.236058
M            -1.361267
N            -0.410836
O            -0.243765
P            -0.292809
Q            -0.228311
R            -0.426401
S            -0.651585
T            -0.626124
dtype: float64

'''