import os
import sys
import pickle
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


BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20/'


@wrap_non_picklable_objects
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
def get_estimate_from_posterior(each_posterior,index_to_node,BASE_PATH,treatment,outcome):
    df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

    graph_sample = nx.from_numpy_array(each_posterior,create_using=nx.DiGraph)
    if nx.is_directed_acyclic_graph(graph_sample): # Check if it is an acyclic DAG
        graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
        estimate = get_causal_estimate(graph_sample_relabeled,df,treatment,outcome)
        return estimate
    else:
        return None    


if __name__=="__main__":
    #print(f"sys argv: {sys.argv}")
    #sys_args = sys.argv[1].split(' ')
    sys_args = sys.argv[1:]
    print(f"sys_args: {sys_args}")

    baseline_to_use = sys_args[0]
    SEED_TO_USE = []

    treatment = str(sys_args[2])
    outcome = str(sys_args[3])

    seed_number = int(sys_args[1])
    if seed_number != 0:
        SEED_TO_USE = [i for i in range(seed_number-5,seed_number)]

    if treatment!=outcome:
        for baseline in [baseline_to_use]:
            for seed in SEED_TO_USE:
                BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))
                PATH_TO_SAVE_TRUE_ATE_ESTIMATES = os.path.join(BASE_PATH,'variable_ates')
                os.makedirs(PATH_TO_SAVE_TRUE_ATE_ESTIMATES,exist_ok=True)

                csv_file_for_ate = os.path.join(PATH_TO_SAVE_TRUE_ATE_ESTIMATES,f'true_ate_estimates_{treatment}_{outcome}.csv')
                if not os.path.exists(csv_file_for_ate):

                    if not os.path.exists(BASE_PATH):
                        continue
                    graph_path = os.path.join(BASE_PATH,'graph.pkl') 
                    with open(graph_path,'rb') as fl:
                        graph = pl.load(fl)


                    # Get posterior
                    count=0
                    posterior_file_path = os.path.join(BASE_PATH,'true_mec_dags.pkl')
                    
                    if not os.path.isfile(posterior_file_path):
                        continue
                    
                    with open(posterior_file_path,'rb') as f:
                        true_posterior = pickle.load(f)

                    #breakpoint()
                    #posterior = np.load(posterior_file_path)[:10,:,:] # for debugging
                    
                    PATH_TO_SAVE_TRUE_ATE_ESTIMATES = os.path.join(BASE_PATH,'variable_ates')
                    os.makedirs(PATH_TO_SAVE_TRUE_ATE_ESTIMATES,exist_ok=True)

                    results = Parallel(n_jobs=len(os.sched_getaffinity(0)))(
                            delayed(get_estimate_from_posterior)(true_posterior[i],index_to_node,BASE_PATH,treatment,outcome)
                            for i in range(len(true_posterior))
                            )
                    causal_estimates = np.asarray(results)
                    causal_estimates_list = causal_estimates.tolist()
                    length_causal_estimates = len(causal_estimates_list)

                    
                    ates_ = causal_estimates_list
                    baselines_ = [baseline for i in range(length_causal_estimates)]
                    seeds_ = [seed for i in range(length_causal_estimates)]
                    treatments = [treatment for i in range(length_causal_estimates)]
                    outcomes = [outcome for i in range(length_causal_estimates)]

                    if len(baselines_)!=0 and len(ates_)!=0 and len(seeds_)!=0:        
                        df = pd.DataFrame({'baselines':baselines_,'true_ates':ates_,'seeds':seeds_,'treatments':treatments,'outcomes':outcomes})
                        df.to_csv(csv_file_for_ate,index=False)
                    else:
                        print("List was empty so nothing was done...")
    print(f'ALL DONE for seed: {SEED_TO_USE}') 