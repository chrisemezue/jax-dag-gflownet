import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from testgr import calculate_squared_diff,calculate_rmse

FOLDER = '/home/mila/c/chris.emezue/jax-dag-gflownet/ate_estimates_sc'

files = [os.path.join(FOLDER,f.name) for f in os.scandir(FOLDER)]


df_files = pd.concat([pd.read_csv(f) for f in files])


fig, ax = plt.subplots()

#import pdb;pdb.set_trace()
g = sns.FacetGrid(df_files,  row="cases")
ax = g.map_dataframe(sns.violinplot, y="rmse",x="baselines")
#ax = sns.boxplot(y="rmse",x="baselines",data=df_files)
#import pdb;pdb.set_trace()
#ax.set_titles('RMSE-ATE for baselines')
plt.xticks(rotation=90)
plt.tight_layout()
ax.savefig('rmse_ate_estimates_BOXPLOT_ALL_20')

'''
all_baselines = []
all_scores=[]

FOLDER2 = '/home/mila/c/chris.emezue/scratch/ate_estimates'
for baseline_to_use in ["bcdnets", "bootstrap_ges", "bootstrap_pc", "dag_gflownet" ,"dibs", "gadget", "mc3"]:
    for seed in range(0,26,5): 
        try:
            causal_estimates = np.load(f'/home/mila/c/chris.emezue/scratch/ate_estimates2/{baseline_to_use}_{seed}_ate_estimates.npy')
            #causal_estimates = np.full(posterior.shape[0], fill_value=1) 
            true_causal_estimates =np.load(f'/home/mila/c/chris.emezue/scratch/ate_estimates2/true_{baseline_to_use}_{seed}_ate_estimates.npy')
            #rmse_value = calculate_squared_diff(causal_estimates,true_causal_estimates)
            rmse_value = calculate_rmse(causal_estimates,true_causal_estimates)
            #import pdb;pdb.set_trace()
            rmse_values_list = rmse_value.tolist() if isinstance(rmse_value.tolist(),list) else [rmse_value.tolist()]
            all_scores.extend(rmse_values_list)
            all_baselines.extend([baseline_to_use for i in rmse_values_list])
        except Exception:
            continue




fig, ax = plt.subplots()



ax = sns.boxplot(y=all_scores,x=all_baselines)
ax.set_title('RMSE of ATE for baselines ')
plt.xticks(rotation=90)
plt.tight_layout()
fig.savefig('RMSE_2_estimates_all_rmse_final')
'''
print('ALL DONE')