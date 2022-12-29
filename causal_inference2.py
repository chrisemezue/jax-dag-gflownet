import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from dowhy import CausalModel
import pickle as pl
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects
from typing import Optional

set_loky_pickler('pickle5') #https://joblib.readthedocs.io/en/latest/auto_examples/serialization_and_wrappers.html

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


# New idea
# Choose two variables apriori: treatment and effect (X & Y)
# We are interested in X -> Y | E[Y|do(X)]
# for each baseline
# look at its groundtruth graph
    # find all paths from X to Y
    # get the true coefficients of the edges
    # path tracing to find the effect of X-> Y

# for each posterior
    # transform it into a graph
    # find all paths from X to Y
    # regress using the data to find the estimated coefficients of the edges
    # path tracing to find the effect of X-> Y



def find_all_paths(graph,variable_x=0,variable_y=3):
    # Given the graph, the source and target variables, find all paths from source to target

    paths = nx.all_simple_paths(graph, source=variable_x, target=variable_y)
    #print(list(paths)) --if you want the list of the paths: [[0, 1, 3], [0, 2, 3], [0, 3]]
    for path in map(nx.utils.pairwise, paths): # each path as the corresponding list of edges
        print(list(path))

    # find all pairwise paths
    # find the weights => [(A,B,0.5),(H,J,-0.4)...]
    # save above in JSON



    # Find all the variables involved in path from X->Y. these are our variables of interest
    # Only do regression involving these variables
    # Get the regression weights.



def get_true_edge_coefficients(graph):
    # Given the true graph that has been unpickled, we want to get all its edge coefficients
    with open('graph.pkl','rb') as f:
        graph = pl.load(f)

def convert_posterior_to_graph(posterior,index_to_node):
    graph_sample = nx.from_numpy_array(posterior,create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph_sample): # Check it is acyclic DAG
        return None
    graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
    return graph_sample_relabeled
   

@wrap_non_picklable_objects
def get_causal_estimate(graph,df):
    model = CausalModel(data=df, treatment=['K'],outcome='M',graph=nx.DiGraph(graph),use_graph_as_is=True)
    # II. Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(identified_estimand,
            target_units='ate',
            control_value=0,
            treatment_value=1,
            method_name="backdoor.linear_regression",
            test_significance=False,confidence_intervals=False)
    causal_estimate = causal_estimate_reg.value
    return causal_estimate

@wrap_non_picklable_objects
def get_estimate_from_posterior(each_posterior,index_to_node,BASE_PATH):
    df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

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
    # Remove `None` values from the RMSE calculation: encountered it in `dibs`
    mask = a!=None
    a_ = a[mask]
    b_ = b[mask]
    ####################
    return np.sqrt(np.mean(np.square(np.subtract(a_, b_)), axis=axis))


def calculate_squared_diff(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the squared difference between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.square(np.subtract(a, b))


if __name__=="__main__":
    baseline_to_use = sys.argv[1]
    SEED_TO_USE = []

    seed_number = int(sys.argv[2])
    if seed_number != 0:
        SEED_TO_USE = [i for i in range(seed_number-5,seed_number)]

    for baseline in [baseline_to_use]:
        for seed in SEED_TO_USE:
            BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

            if not os.path.exists(BASE_PATH):
                continue

            with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
                graph = pl.load(fl)


            # Get posterior
            count=0
            posterior_file_path = os.path.join(BASE_PATH,'posterior.npy')
            if not os.path.isfile(posterior_file_path):
                continue
            posterior = np.load(posterior_file_path)
            #posterior = np.load(posterior_file_path)[:10,:,:] # for debugging
            
            # Without parallelization
            #causal_estimates = np.array([get_estimate_from_posterior(posterior[i,:,:],index_to_node,BASE_PATH) for i in range(posterior.shape[0])])

            results = Parallel(n_jobs=len(os.sched_getaffinity(0)))(
                    delayed(get_estimate_from_posterior)(posterior[i,:,:],index_to_node,BASE_PATH)
                    for i in range(posterior.shape[0])
                    )
            causal_estimates = np.asarray(results)
            df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))


            true_estimate = get_causal_estimate(graph,df)

            with open(f'/home/mila/c/chris.emezue/scratch/ate_estimates2/{baseline_to_use}_{seed}_ate_estimates.npy', 'wb') as fl:
                np.save(fl,causal_estimates)
            true_causal_estimates = np.full(causal_estimates.shape, fill_value=true_estimate)
            with open(f'/home/mila/c/chris.emezue/scratch/ate_estimates2/true_{baseline_to_use}_{seed}_ate_estimates.npy', 'wb') as fl:
                np.save(fl,true_causal_estimates)
            rmse_value = calculate_rmse(causal_estimates,true_causal_estimates)
            
            baselines_.append(baseline)
            rmses_.append(rmse_value)
            seeds_.append(seed)

    if len(baselines_)!=0 and len(rmses_)!=0 and len(seeds_)!=0:        
        df = pd.DataFrame({'baselines':baselines_,'rmse':rmses_,'seeds':seeds_})
        df.to_csv(f'ate_estimates2/{baseline_to_use}_{SEED_TO_USE}_ate_estimates.csv',index=False)
    else:
        print("List was empty so nothing was done")
    print(f'ALL DONE for seeds: {SEED_TO_USE}') 