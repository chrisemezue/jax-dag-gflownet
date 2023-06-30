import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pl
from tqdm import tqdm
from eval_utils import get_distribution_metrics,get_ate_precision_recall,regroup_close_values

def read_pickle_path(path_):
    with open(path_,'rb') as f:
        return pl.load(f)

#FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/sachs_obs'
FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss100'

BASELINES = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3','dag-gfn']

if 'sachs' in FOLDER:
    SEEDS = [0]
else:
    SEEDS = [i for i in range(26)]


dfs_list = []


for baseline in BASELINES:
    for seed in SEEDS:
        
        baseline_seed_folder = os.path.join(os.path.join(FOLDER,baseline),str(seed))
        kde_folder = os.path.join(baseline_seed_folder,'kde')
        if os.path.exists(kde_folder):
            pds_paths = [os.path.join(kde_folder,f.name) for f in os.scandir(kde_folder) if f.name.endswith('.csv') and os.path.exists(os.path.join(kde_folder,f.name))]
            #pds_paths = [p for p in pds_paths if os.path.exists(p)]
            pds = [pd.read_csv(f) for f in pds_paths]
            dfs_list.extend(pds)

df = pd.concat(dfs_list)

# breakpoint()
# # convert df to long format
# # plot bar chart
# df_prec = pd.melt(df, id_vars='baselines', value_vars=['precision','precision_0','precision_025','precision_05'],var_name='precisions', value_name='amount')

# df_rec = pd.melt(df, id_vars='baselines', value_vars=['recall','recall_0','recall_025','recall_05'],var_name='recalls', value_name='amount')

# fig, ax = plt.subplots()
# df_prec_mean= df_prec.groupby(['baselines','precisions']).mean().reset_index()
# ax = sns.catplot(x = 'baselines', y='amount', hue = 'precisions',data=df_prec_mean,kind='bar')
# ax.set_xticklabels(ax.ax.get_xticklabels(), rotation = 45, ha="right")

# #ax.set_titles('Different precision averages')
# ax.savefig('metric_precisions_baselines_20.png')

# ## for recall

# fig2, ax2 = plt.subplots()
# df_rec_mean= df_rec.groupby(['baselines','recalls']).mean().reset_index()
# ax2 = sns.catplot(x = 'baselines', y='amount', hue = 'recalls',data=df_rec_mean,kind='bar')

# ax2.set_xticklabels(ax2.ax.get_xticklabels(), rotation = 45, ha="right")
# #ax2.set_titles('Different recall averages')
# ax2.savefig('metric_recall_baselines_20.png')


# ['precision','recall','precision_0','recall_0','precision_025','recall_025','precision_05','recall_05']
#breakpoint()
df2 = df.groupby(['baselines']).mean().reset_index()
bb= df2['baselines'].values.tolist()
w =df2['wasserstein'].values.tolist()
pr = df2['precision'].values.tolist()
pr_05 = df2['precision_05'].values.tolist()
rec = df2['recall'].values.tolist()
rec_05 = df2['recall_05'].values.tolist()

df3 = df.groupby(['baselines']).std().reset_index()
w2 =df3['wasserstein'].values.tolist()
pr2 = df3['precision'].values.tolist()
rec2 = df3['recall'].values.tolist()
pr2_05 = df3['precision_05'].values.tolist()
rec2_05 = df3['recall_05'].values.tolist()

s = ''
for i in range(len(bb)):
    s = s+f'{bb[i]} & {w[i]:.3f} \pm {w2[i]:.3f} & {pr[i]:.2f} \pm {pr2[i]:.2f} & {rec[i]:.2f} \pm {rec2[i]:.2f}  & {pr_05[i]:.2f} \pm {pr2_05[i]:.2f} & {rec_05[i]:.2f} \pm {rec2_05[i]:.2f} \\'+'\n'


breakpoint()


def get_number_graphs_in_mec():
    ba,mecs,sed = [],[],[]
    FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/sachs_obs'

    BASELINES = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3','dag-gfn']

    if 'sachs' in FOLDER:
        SEEDS = [0]
    else:
        SEEDS = [i for i in range(26)]
    
    for baseline in BASELINES:
        for seed in SEEDS:
            
            baseline_seed_folder = os.path.join(os.path.join(FOLDER,baseline),str(seed))
            mec_dags_path = os.path.join(baseline_seed_folder,'true_mec_dags.pkl') 
            if os.path.exists(mec_dags_path):
                mec_dags = read_pickle_path(mec_dags_path)
                #breakpoint()
                ba.append(baseline)
                mecs.append(len(mec_dags))
                sed.append(seed)
    
    df = pd.DataFrame({'baseline':ba,'Size of MEC':mecs,'seed':sed})
    breakpoint()
    fig,ax = plt.subplots()
    ax = sns.histplot(data=df, x="Size of MEC")
    fig.savefig('size_of_mec.png')




def get_distribution_of_kdes(df):
    kde_paths = df['kde'].values.tolist()
    kde_bws = [read_pickle_path(k).bandwidth for k in kde_paths]
    return kde_bws


#bws = get_distribution_of_kdes(df)
#breakpoint()


def check_if_true_graph_multimodal(list_):
    # check the len of its set.
    # if it is more than 1, then perhaps multimodal?
    list_set = set(list_)
    if len(list_set) > 1:
        return True
    else:
        return False
def get_plot_multimodal():
    IGNORE = True
    KDE_FOLDER = '/home/mila/c/chris.emezue/scratch/causal_inference/kde_sachs'
    BASEFOLDER  = '/home/mila/c/chris.emezue/jax-dag-gflownet/plots_others'
    SACHS_VARIABLES_LIST = ['Akt' ,'Erk' ,'Jnk' ,'Mek', 'P38' ,'PIP2' ,'PIP3' ,'PKA' ,'PKC', 'Plcg' ,'Raf']
    TEMPLATE = '/home/mila/c/chris.emezue/gflownet_sl/tmp/sachs_obs/dag-gfn/0/variable_ates/true_ate_estimates_{}_{}.csv'
    with tqdm((len(SACHS_VARIABLES_LIST)**2) * len(BASELINES),desc = 'Interesting variables...') as pbar:
        for treatment in SACHS_VARIABLES_LIST:
            for outcome in SACHS_VARIABLES_LIST:
                if treatment=='Raf' and outcome=='Mek':
                #if True:

                    # Get true ATE of it.
                    true_ate_filename = TEMPLATE.format(treatment,outcome)
                    if os.path.exists(true_ate_filename):
                        true_df = pd.read_csv(true_ate_filename)
                        true_ates = true_df['true_ates'].values.tolist()
                        if check_if_true_graph_multimodal(true_ates) or IGNORE:
                            FIG_FOLDER = os.path.join(BASEFOLDER,f'{treatment}-{outcome}')
                            os.makedirs(FIG_FOLDER,exist_ok = True)
                            fig,ax = plt.subplots(3,3,sharex=False,sharey=True)
                            #fig,ax = plt.subplots(1,3,sharex=False,sharey=True,squeeze = False)

                            fig.tight_layout()
                            ax[0,0].hist(true_ates)
                            ax[0,0].set_title('True ATEs',fontdict = {'fontsize': 10})
                            #for i,baseline in enumerate(['bootstrap_pc','dag-gfn']):
                            for i,baseline in enumerate(BASELINES):
                                baseline_kde_filename = os.path.join(KDE_FOLDER,f'{baseline}_0_{treatment}_{outcome}_kde.pkl')
                                if os.path.exists(baseline_kde_filename):
                                    with open(baseline_kde_filename,'rb') as f:
                                        kde = pl.load(f)
                                    ate_preds = np.asarray(kde.tree_.data).squeeze().tolist()
                                    #fig2,ax2 = plt.subplots()
                                    k =int((i+1)/3) 
                                    m = (i+1)%3

                                    #k =0
                                    #m = (i+1)%3

                                    ax[k,m].hist(ate_preds)
                                    prec, rec, _ = get_distribution_metrics(ate_preds,true_ates)
                                    if baseline == 'dag-gfn':
                                        breakpoint()
                                        X = np.linspace(0.001,0.7,50)
                                        metrics = [get_ate_precision_recall(ate_preds,true_ates,m) for m in X]
                                        prec = [m[0] for m in metrics]
                                        rec = [m[1] for m in metrics]
                                        fig2,ax2 = plt.subplots(2,1,squeeze= False, sharex=True,sharey=False)
                                        ax2[0,0].plot(X,prec,color='blue',label='precision')
                                        ax2[1,0].plot(X,rec,color='red',label='recall')
                                        ax2[0,0].legend()
                                        ax2[1,0].legend()
                                        ax2[1,0].set_xlabel('tol')
                                        fig2.tight_layout()
                                        fig2.suptitle(f'DAG-GFN | {treatment} -> {outcome}')
                                        plt.tight_layout()

                                        fig.savefig('prec_rec_plot_pka_pip3.png')
                                        breakpoint()

                                    prec2, rec2, _ = get_ate_precision_recall(ate_preds,true_ates)

                                    ax[k,m].set_title(f'{baseline} | prec: {prec:.2f} ({prec2:.2f}), rec:{rec:.2f} ({rec2:.2f})',fontdict = {'fontsize': 8})
                            ax[2,2].cla()
                            #fig.suptitle(f'ATE distribution for {treatment} -> {outcome} | Sachs')
                            #fig.suptitle(f'ATE distributions for {treatment} -> {outcome}')
                            breakpoint()

                            fig.savefig(os.path.join(FIG_FOLDER,f'ate_histogram.png'))
                    breakpoint()
                pbar.update(1)


def get_true_ate_unique_hist():
    num_unique_ates = []
    BASELINES = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3','dag-gfn']

    BASEFOLDER  = '/home/mila/c/chris.emezue/jax-dag-gflownet/plots_others'
    SACHS_VARIABLES_LIST = ['Akt' ,'Erk' ,'Jnk' ,'Mek', 'P38' ,'PIP2' ,'PIP3' ,'PKA' ,'PKC', 'Plcg' ,'Raf']
    TEMPLATE = '/home/mila/c/chris.emezue/gflownet_sl/tmp/sachs_obs/{}/0/variable_ates/true_ate_estimates_{}_{}.csv'
    with tqdm((len(SACHS_VARIABLES_LIST)**2) * len(BASELINES)) as pbar:
        for treatment in SACHS_VARIABLES_LIST:
            for outcome in SACHS_VARIABLES_LIST:
                #if treatment!=outcome:
                if treatment == 'Erk' and outcome=='Raf':
                    for baseline in BASELINES:
                    #if treatment=='Raf' and outcome=='Mek':
                    #if True:

                        # Get true ATE of it.
                        breakpoint()
                        true_ate_filename = TEMPLATE.format(baseline,treatment,outcome)
                        if os.path.exists(true_ate_filename):
                            true_df = pd.read_csv(true_ate_filename)
                            true_ates = true_df['true_ates'].values.tolist()
                            #breakpoint()
                            true_ates2 = regroup_close_values(true_ates)
                            #unique_true_ates = list(set(true_ates))
                            #num_unique_ates.append(len(unique_true_ates))
                            num_unique_ates.append(len(true_ates2))

                    pbar.update(1)

    fig,ax = plt.subplots()
    breakpoint()
    ax.hist(num_unique_ates)
    ax.set_title('Unique ATEs for Erk -> Raf \n (using regrouping)')
    ax.set_ylabel('Count')
    ax.set_xlabel('# Unique ATEs')
    fig.tight_layout()
    fig.savefig('size_unique_ates_erk_raf.png')

get_true_ate_unique_hist()
#get_plot_multimodal()
#breakpoint()
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
