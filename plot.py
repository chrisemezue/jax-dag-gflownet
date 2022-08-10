import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = '/home/mila/c/chris.emezue/jax-dag-gflownet/ate_estimates'

files = [os.path.join(FOLDER,f.name) for f in os.scandir(FOLDER)]


df_files = pd.concat([pd.read_csv(f) for f in files])


fig, ax = plt.subplots()



ax = sns.boxplot(y="rmse",x="baselines",data=df_files)
ax.set_title('RMSE-ATC for baselines (using one seed)')
plt.xticks(rotation=90)
plt.tight_layout()
fig.savefig('rmse_ate_estimates')






