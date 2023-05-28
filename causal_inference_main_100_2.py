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
from timeTracker import TimeTracker

class Logger:
    def log(self, message):
        #breakpoint()
        print('[TimeTracking] {function_name} - total: {total_time:3.6f} s - partial: {partial_time:3.6f} s'.format(**message))

logger = Logger()
time_tracker = TimeTracker(logger, True)

set_loky_pickler('pickle5') #https://joblib.readthedocs.io/en/latest/auto_examples/serialization_and_wrappers.html

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 


@wrap_non_picklable_objects
#@time_tracker.time_track()
def get_causal_estimate(graph,df,treatment,outcome):
    model = CausalModel(data=df, treatment=[treatment],outcome=outcome,graph=nx.DiGraph(graph),use_graph_as_is=True)
    # II. Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(identified_estimand,
            target_units='ate',
            control_value=0.0,
            treatment_value=1.0,
            method_name="backdoor.linear_regression",
            test_significance=False,confidence_intervals=False)
    causal_estimate = causal_estimate_reg.value
    return causal_estimate

@wrap_non_picklable_objects
#@time_tracker.time_track()
def get_estimate_from_posterior(each_posterior,index_to_node,BASE_PATH,treatment,outcome):
    df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

    graph_sample = nx.from_numpy_array(each_posterior,create_using=nx.DiGraph)
    if nx.is_directed_acyclic_graph(graph_sample): # Check if it is an acyclic DAG
        graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
        estimate = get_causal_estimate(graph_sample_relabeled,df,treatment,outcome)
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


# Calculate only the ATE (no need for RMSE) and save it.
# Define a format for saving them ==> to enable easier collation for downstream analysis.
# Loop through all combinations of target and effect variable

#@time_tracker.time_track()
def main():
    BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss100/'
    ATE_DATAFRAME_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main_100' 

    sys_args = sys.argv[1].split(' ') # if using SLURM
    if len(sys_args)==1:
        sys_args = sys.argv[1:] # if using bash
    #sys_args = sys.argv[1:] #if not using SLURM
    print(sys_args)
    baseline_to_use = sys_args[0]
    SEED_TO_USE = []
    baselines_ = []
    ates_ = []
    seeds_ = []
    treatments = []
    outcomes = []
    true_graph_paths = []
    true_estimates = []
    times = []

    treatment = str(sys_args[2])
    outcome = str(sys_args[3])

    os.makedirs(ATE_DATAFRAME_FOLDER,exist_ok=True)


    seed_number = int(sys_args[1])
    SEED_TO_USE = [i for i in range(seed_number-5,seed_number)]

    ate_estimate_csv_filename = os.path.join(ATE_DATAFRAME_FOLDER,f'{baseline_to_use}_{SEED_TO_USE}_{treatment}_{outcome}_ate_estimates.csv')
    
    if not os.path.exists(ate_estimate_csv_filename):
        print(f'ATE file already exists. Skipping! \n {ate_estimate_csv_filename}')
    else:
        if treatment!=outcome:
            for baseline in [baseline_to_use]:
                for seed in SEED_TO_USE:
                    if seed ==14:
                        BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

                        if not os.path.exists(BASE_PATH):
                            continue
                        graph_path = os.path.join(BASE_PATH,'graph.pkl') 
                        with open(graph_path,'rb') as fl:
                            graph = pl.load(fl)


                        # Get posterior
                        count=0
                        posterior_file_path = os.path.join(BASE_PATH,'posterior.npy') if baseline!='dag-gfn' else os.path.join(BASE_PATH,'posterior_estimate.npy') 
                        
                        
                        if not os.path.isfile(posterior_file_path):
                            continue
                        posterior = np.load(posterior_file_path)
                        #posterior = np.load(posterior_file_path)[:10,:,:] # for debugging
                        
                        # Without parallelization
                        #causal_estimates = np.array([get_estimate_from_posterior(posterior[i,:,:],index_to_node,BASE_PATH) for i in range(posterior.shape[0])])
                        st_time = time.time()
                        #results = Parallel(n_jobs=len(os.sched_getaffinity(0)))(
                        #        delayed(get_estimate_from_posterior)(posterior[i,:,:],index_to_node,BASE_PATH,treatment,outcome)
                        #        for i in range(posterior.shape[0])
                        #        )

                        # without parallel
                        results = [get_estimate_from_posterior(posterior[i,:,:],index_to_node,BASE_PATH,treatment,outcome) for i in range(posterior.shape[0])]

                        times.append(time.time() - st_time)
                        breakpoint()
                        causal_estimates = np.asarray(results)
                        causal_estimates_list = causal_estimates.tolist()
                        length_causal_estimates = len(causal_estimates_list)
                        #breakpoint()
                        df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))
                        true_estimate = get_causal_estimate(graph,df,treatment,outcome)
                        
                        ates_.extend(causal_estimates_list)
                        baselines_.extend([baseline for i in range(length_causal_estimates)])
                        seeds_.extend([seed for i in range(length_causal_estimates)])
                        treatments.extend([treatment for i in range(length_causal_estimates)])
                        outcomes.extend([outcome for i in range(length_causal_estimates)])
                        true_graph_paths.extend([graph_path for i in range(length_causal_estimates)])
                        true_estimates.extend([true_estimate for i in range(length_causal_estimates)])

        if len(baselines_)!=0 and len(ates_)!=0 and len(seeds_)!=0:        
            df = pd.DataFrame({'baselines':baselines_,'ates':ates_,'seeds':seeds_,'treatments':treatments,'outcomes':outcomes,'true_graph_paths':true_graph_paths,'true_ATE':true_estimates})
            df.to_csv(ate_estimate_csv_filename,index=False)
        else:
            print("List was empty so nothing was done...")
        print(f'ALL DONE for seeds: {SEED_TO_USE}') 


    breakpoint()


if __name__=="__main__":
    main()