import os
import sys
import subprocess
from tqdm import tqdm

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 

#BASELINE_FOLDER = '/home/mila/c/chris.emezue/scratch/baselines/'
BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss100/'
ATE_DATAFRAME_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main_100' 



VARIABLES = list(node_to_index.keys())



if __name__=="__main__":

    #baseline_to_use = sys_args[0]
    SEED_TO_USE = []
    baselines_ = []
    ates_ = []
    seeds_ = []
    treatments = []
    outcomes = []
    true_graph_paths = []
    true_estimates = []
    not_found = []
    undone = 0

    with tqdm(25*len(VARIABLES)*len(VARIABLES)*7,desc='Checking undone ATE...',bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as pbar:
        for seed in range(26):
            for treatment in VARIABLES:
                for outcome in VARIABLES:
                    for baseline in ["bcdnets", "bootstrap_ges", "bootstrap_pc" ,"dibs" ,"gadget", "mc3", "dag-gfn"]:

                        if treatment!=outcome:
                            #for baseline in [baseline_to_use]:
                            #for baseline in ["bcdnets", "bootstrap_ges", "bootstrap_pc" ,"dag_gflownet" ,"dibs" ,"gadget", "mc3", "dag-gfn"]   
                            BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

                            if not os.path.exists(BASE_PATH):
                                continue

                            cmd_str = f'ls "{ATE_DATAFRAME_FOLDER}" | grep -P "{baseline}_.*[^0-9]{seed}[,\]].*_{treatment}_{outcome}.*.csv"'
                            #subprocess.run(cmd_str, shell=True)
                            try:
                                return_code = subprocess.check_output(cmd_str, shell=True)
                            except subprocess.CalledProcessError as err:
                                #breakpoint()
                                undone+=1 

                                csv_filename = f"{baseline}-{seed}-{treatment}-{outcome}"
                                #csv_filename= os.path.join(ATE_DATAFRAME_FOLDER,f'{baseline}_{SEED_TO_USE}_{treatment}_{outcome}_ate_estimates.csv')
                                #if not os.path.exists(csv_filename):
                                not_found.append(csv_filename)

                        pbar.update(1)

    with open('/home/mila/c/chris.emezue/jax-dag-gflownet/undone_ate_experiments.txt','w+') as f:
        for nf in not_found:
            f.write(nf+'\n')
    print(f'Found {len(not_found)} undone files.')