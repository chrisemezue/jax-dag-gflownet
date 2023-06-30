import os
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm



if __name__=="__main__":
    BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/sachs_obs'
    print(BASELINE_FOLDER)

    baseline_to_use = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3','dag-gfn']
    #baseline_to_use = ['dag-gfn']

    SEED_TO_USE = [i for i in range(26)]
    variance=[]
    cyclic_arr = []
    with tqdm(total = len(baseline_to_use)*len(SEED_TO_USE)) as pbar:
        for baseline in baseline_to_use:
            for seed in SEED_TO_USE:
                baseline_path = os.path.join(BASELINE_FOLDER,baseline)
                BASE_PATH = os.path.join(baseline_path,str(seed))

                if not os.path.exists(BASE_PATH):
                    continue
                posterior_file_path = os.path.join(BASE_PATH,'posterior.npy') if baseline!='dag-gfn' else os.path.join(BASE_PATH,'posterior_estimate.npy') 
                                  
                if not os.path.isfile(posterior_file_path):
                    continue
                posterior = np.load(posterior_file_path)

                is_cyclic = [1 if not nx.is_directed_acyclic_graph(nx.from_numpy_array(posterior[i,:,:],create_using=nx.DiGraph)) else 0 for i in range(posterior.shape[0])]
                #breakpoint()
                mean_cyclic = sum(is_cyclic)
                cyclic_arr.append(mean_cyclic)

                pbar.update(1)
            # print % cyclic
            #if baseline=='dibs':
            #    breakpoint()
            print(f'Total cyclic DAGs for {baseline}: {sum(cyclic_arr)}')
            cyclic_arr = []

    print('ALL DONE')
