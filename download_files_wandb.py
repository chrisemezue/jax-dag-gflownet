import wandb
import os
import numpy as np
from tqdm.auto import tqdm

def get_files(runs, filename, root_folder):
    api = wandb.Api()

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    basename, ext = os.path.splitext(os.path.join(root_folder, filename))

    for run in runs:
        if run.state == 'finished': # Only take finished runs (prevents us from taking crashed runs which may not have all the files we need).
            folder_for_run = os.path.join(root_folder,run.name)
            os.makedirs(folder_for_run,exist_ok=True)

            if isinstance(run, str):
                run = api.run(run)
            try:
                
                run.file(filename).download(root=folder_for_run)   
                #print(f'Downloaded `{"/".join(run.path)}`: {folder_for_run}/{filename}')

                # do the extra step for `posterior_estimate.npy` file
                if filename=='posterior_estimate.npy':
                    with open(f'{folder_for_run}/{filename}', 'rb') as f:
                        orders = np.load(f)
                        adjacencies = (orders >= 0).astype(np.int_)
                    #breakpoint()
                    np.save(f'{folder_for_run}/{filename}',adjacencies)
                    #print(f'Adjusted the posterior to get DAG and saved it.')
                    

            except wandb.errors.CommError:
                print(f'Could not download file: {filename} from {run.name} | id: {run.id}')
                continue


if __name__ == '__main__':
    #ROOT_FOLDER_DAG_GFN = '/home/mila/c/chris.emezue/scratch/test_wandb/dag-gfn' # for debugging
    ROOT_FOLDER_DAG_GFN= '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss5_100/dag-gfn' # the real deal, be careful


    # Download from runs selected with a tag
    api = wandb.Api()
    runs = api.runs('tristandeleu_mila_01/gflownet-bayesian-structure-learning', 
        filters={'tags': {'$in': ['causal_inference_main_5_100']}}
    )
    FILES = ['posterior_estimate.npy','data.csv','graph.pkl']
    for file in tqdm(FILES, desc="Downloading files..."):
        get_files(runs, file, ROOT_FOLDER_DAG_GFN)
