import os
import sys

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 

#BASELINE_FOLDER = '/home/mila/c/chris.emezue/scratch/baselines/'
BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20/'


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
    not_found =[]

    for seed_number in range(5,26,5):
        for treatment in VARIABLES:
            for outcome in VARIABLES:
                for baseline in ["bcdnets", "bootstrap_ges", "bootstrap_pc" ,"dibs" ,"gadget", "mc3", "dag-gfn"]:



                    #treatment = str(sys_args[2])
                    #outcome = str(sys_args[3])

                    #SAVE_ATE_ESTIMATES_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main'
                    ATE_DATAFRAME_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main_20' 
                    #os.makedirs(SAVE_ATE_ESTIMATES_FOLDER,exist_ok=True)
                    #os.makedirs(ATE_DATAFRAME_FOLDER,exist_ok=True)

                    #seed_number = int(sys_args[1])
                    if seed_number != 0:
                        SEED_TO_USE = [i for i in range(seed_number-5,seed_number)]

                    if treatment!=outcome:
                        #for baseline in [baseline_to_use]:
                        #for baseline in ["bcdnets", "bootstrap_ges", "bootstrap_pc" ,"dag_gflownet" ,"dibs" ,"gadget", "mc3", "dag-gfn"]   
                        for seed in SEED_TO_USE:
                            BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

                            if not os.path.exists(BASE_PATH):
                                continue
                            #graph_path = os.path.join(BASE_PATH,'graph.pkl') 
                            #with open(graph_path,'rb') as fl:
                            #    graph = pl.load(fl)


                            # Get posterior
                            count=0
                            posterior_file_path = os.path.join(BASE_PATH,'posterior.npy') if baseline!='dag-gfn' else os.path.join(BASE_PATH,'posterior_estimate.npy') 
                            
                            
                            if not os.path.isfile(posterior_file_path):
                                continue
                            #posterior = np.load(posterior_file_path)
                            #posterior = np.load(posterior_file_path)[:10,:,:] # for debugging
                            
                            # Without parallelization
                            #causal_estimates = np.array([get_estimate_from_posterior(posterior[i,:,:],index_to_node,BASE_PATH) for i in range(posterior.shape[0])])

                            #results = Parallel(n_jobs=len(os.sched_getaffinity(0)))(
                            #        delayed(get_estimate_from_posterior)(posterior[i,:,:],index_to_node,BASE_PATH,treatment,outcome)
                            #        for i in range(posterior.shape[0])
                            #        )
                            #causal_estimates = np.asarray(results)
                            #causal_estimates_list = causal_estimates.tolist()
                            #length_causal_estimates = len(causal_estimates_list)
                            #breakpoint()
                            #df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))
                            #true_estimate = get_causal_estimate(graph,df,treatment,outcome)

                            # Save predicted causal estimates
                            #with open(os.path.join(SAVE_ATE_ESTIMATES_FOLDER,f'{baseline_to_use}_{seed}_ate_estimates.npy'), 'wb') as fl:
                            #    np.save(fl,causal_estimates)
                            
                            # Save true causal estimates
                            #true_causal_estimates = np.full(causal_estimates.shape, fill_value=true_estimate)
                            #with open(os.path.join(SAVE_ATE_ESTIMATES_FOLDER,f'true_{baseline_to_use}_{seed}_ate_estimates.npy'), 'wb') as fl:
                            #    np.save(fl,true_causal_estimates)
                            
                            #ates_.extend(causal_estimates_list)
                            #baselines_.extend([baseline for i in range(length_causal_estimates)])
                            #seeds_.extend([seed for i in range(length_causal_estimates)])
                            #treatments.extend([treatment for i in range(length_causal_estimates)])
                            #outcomes.extend([outcome for i in range(length_causal_estimates)])
                            #true_graph_paths.extend([graph_path for i in range(length_causal_estimates)])
                            #true_estimates.extend([true_estimate for i in range(length_causal_estimates)])

                        csv_filename= os.path.join(ATE_DATAFRAME_FOLDER,f'{baseline}_{SEED_TO_USE}_{treatment}_{outcome}_ate_estimates.csv')
                        if not os.path.exists(csv_filename):
                            not_found.append(csv_filename)

    #if len(baselines_)!=0 and len(ates_)!=0 and len(seeds_)!=0:        
    #    df = pd.DataFrame({'baselines':baselines_,'ates':ates_,'seeds':seeds_,'treatments':treatments,'outcomes':outcomes,'true_graph_paths':true_graph_paths,'true_ATE':true_estimates})
    #    df.to_csv(os.path.join(ATE_DATAFRAME_FOLDER,f'{baseline_to_use}_{SEED_TO_USE}_{treatment}_{outcome}_ate_estimates.csv'),index=False)
    #else:
    #    print("List was empty so nothing was done...")
    #print(f'ALL DONE for seeds: {SEED_TO_USE}') 



    with open('/home/mila/c/chris.emezue/jax-dag-gflownet/undone_ate_experiments.txt','w+') as f:
        for nf in not_found:
            f.write(nf+'\n')
    print(f'Found {len(not_found)} undone files.')

    # Tip: save the path to the true graph instead.    
