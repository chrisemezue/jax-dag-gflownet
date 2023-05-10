import os
import pickle as pl
import pandas as pd

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 

VARIABLES = [k for k in list(node_to_index.keys())]

BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20'
ATE_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main_20'

ate_csvs = [os.path.join(ATE_FOLDER,f.name) for f in os.scandir(ATE_FOLDER) if f.name.endswith('csv')]
ate_dataframes = [pd.read_csv(f) for f in ate_csvs]
ate_dataframe_concatenated = pd.concat(ate_dataframes) # this is the one main dataframe.


baseline_to_use = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3']
#baseline_to_use = ['dag-gfn']

SEED_TO_USE = [i for i in range(26)]
variance=[]
with tqdm(total = len(baseline_to_use)*len(SEED_TO_USE)) as pbar:
    for baseline in baseline_to_use:
        for seed in SEED_TO_USE:
            baseline_path = os.path.join(BASELINE_FOLDER,baseline)
            BASE_PATH = os.path.join(baseline_path,str(seed))

            if not os.path.exists(BASE_PATH):
                continue

            for treatment in VARIABLES:
                for outcome in VARIABLES:
                    if treatment!=outcome:
                        ate_dataframe_concatenated
                        # filter dataframe based on seed, treatment, outcome, baseline.
                        ate_of_interest = ate_dataframe_concatenated.query(f'baselines=="{baseline}" & seeds=="{seed}" & treatments=="{treatment}" & outcomes=="{outcome}"')
                        estimate_ate_samples = ate_of_interest['ates'].values.tolist()


                        with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
                            graph = pl.load(fl)
                        graph = nx.DiGraph(graph)
                        
                        cpdag = get_CPDAG(graph)

                        with open(os.path.join(BASE_PATH,'true_cpdag.pkl'),'wb+') as f:
                            pl.dump(cpdag,f)


# given a seed
# give a baseline. given two variables. 
#  get the list of ATE for those variables across all seeds -> A
#  get_kde for A -> K.
#  get list of ATE for truth graph DAGs across all seeds for that baseline => U
#  get kde_ll (U) given K. Save this as pickle. also save the min, max and mean in a dict. We also want to save



# Things to save
        # K
        # kde_ll(U)