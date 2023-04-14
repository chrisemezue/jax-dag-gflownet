import os
import sys
import networkx as nx
import pickle as pl
import json
from tqdm import tqdm



def get_weights(graph):
    weight_list = []
    for cpd in graph.cpds:
        node = cpd.variable
        for source_node, weight in zip(cpd.variables,cpd.mean.tolist()):
            if source_node!=node:
                weight_list.append((source_node,node,weight))
    return weight_list

def get_variances(graph):
    weight_list = []
    for cpd in graph.cpds:
        weight_list.append(cpd.variance)
    return weight_list


BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20'

    


if __name__=="__main__":
    #baseline_to_use = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3']
    baseline_to_use = ['dag-gfn']

    SEED_TO_USE = [i for i in range(26)]
    variance=[]
    with tqdm(total = len(baseline_to_use)*len(SEED_TO_USE)) as pbar:
        for baseline in baseline_to_use:
            for seed in SEED_TO_USE:
                baseline_path = os.path.join(BASELINE_FOLDER,baseline)
                BASE_PATH = os.path.join(baseline_path,str(seed))

                if not os.path.exists(BASE_PATH):
                    continue

                with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
                    graph = pl.load(fl)
                
                edge_weights = get_weights(graph)

                with open(os.path.join(BASE_PATH,'true_edge_weights.json'),'w+') as f:
                    json.dump(edge_weights,f)
                
                pbar.update(1)

    print('ALL DONE')


# This codebase extracts the edge weights from the cpds of the true graph.
# This only pertains to the true graph.