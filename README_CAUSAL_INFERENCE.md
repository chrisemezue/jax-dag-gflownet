## Benchmarking causal discovery methods for the downstream task of cause effect estimation (ATE)

- Working document is [here](https://www.notion.so/chrisemezue/My-Mila-Project-29df7ef1d7954505abae8ab5361b2410)


## What to do
There are two parts to this project: the causal discovery pipeline and the causal inference (ATE) pipeline

### 1️⃣ Causal Discovery pipeline
This involves training the causal discovery baseline models. 
Some important variables here are
- `num_variables`
- `num_samples`

1. Need to cd into `/home/mila/c/chris.emezue/gflownet_sl`. This is the codebase for all causal discovery.
2. The python file `eval_baselines_lingauss.py` handles causal discovery for `gadget`, `mc3`, `dibs`, and `bcdnets`.
3. Therefore to do causal discovery for the above baselines, simply run: `bash run.sh`. It has the `run_baselines_lingauss20.sh` which handles the CD part.

    Inside `run_baselines_lingauss20.sh` you specify the number of variables, edges as well as  the `folder` -- this is an important parameter because it is where the experimental graph and data as well as the learned posterior will be stored. You will need this for phase 2️⃣.

4. **Running for `bootstrap_pc`, `bootstrap_ges`**: For these two bootstrap-based baselines, we have separated sh files for them -- `run_lingauss_bootstrap_pc.sh` for `bootstrap_pc`  and `run_lingauss_bootstrap_ges.sh` for `bootstrap_ges`. Again, here you can control the number of variables, edges as well as  the `folder`
5. **Running for dag-gfn**: The bash file called [`job.sh`](https://github.com/chrisemezue/gflownet_sl/blob/chris/ci/job.sh) handles it. Under the hood, it uses the `main.py` python file and stores all required files on [WANDB](https://wandb.ai/tristandeleu_mila_01/gflownet-bayesian-structure-learning/table?workspace=user-chrisemezue). Running `job.sh` already enables parallelization by running the different seeds on separate cluster nodes.

    For WANDB, I use this tag structure `tags=[f'causal_inference_{args.num_samples}'] to distinguish the different run results I am running for the causal inference. Another thing I did was to configure the name of the wandb instance to be the seed number. All these were done to enable easy downloading of the required files for causal inference in phase 2️⃣.
    
    You run this experiment by doing `sbatch job.sh NUM_SAMPLES`.


### 2️⃣ Causal Inference pipeline

Now that we have done causal discovery, and have the true graph, observational data and learned posterior for each of the baselines, it is time to do causal inference.

For now, we are mostly concerned with average treatment effect (ATE). I explain it in some detail [here](https://www.notion.so/chrisemezue/My-Mila-Project-29df7ef1d7954505abae8ab5361b2410?pvs=4#4e2ca9d22807470c80679d726652a679).

Our codebase for this can be found [here on Github](https://github.com/chrisemezue/jax-dag-gflownet/tree/master).

1. First need to cd into our codebase at `/home/mila/c/chris.emezue/jax-dag-gflownet`.
2. Before doing any causal inference, there are a number of important preliminary steps to take such that the whole setup is ready when we want it.
3. **Extract and compile relevant files**: It is important to extract and compile the 1) learned posterior, 2) true graph and 3) observational data for each of the baselines and for each of the seeds. These files are denoted as `FILES = ['posterior_estimate.npy','data.csv','graph.pkl']`.

    Remember the `folder` variable from phase 1️⃣ ? This is where it is needed. In this part of the experiment, we will set it to the variable `BASELINE_FOLDER`. For smoothness, we should keep all the files under one and the same `BASELINE_FOLDER` and in the same format. 

    So for the following baselines, we only need to set their `folder` variable to anywhere we see `BASELINE_FOLDER` in the causal inference code: `['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3']`. 

    For dag-gfn, since the files are stored on WANDB, we need to retrieve them. That is what the file `download_files_wandb.py` does. You should set the `ROOT_FOLDER_DAG_GFN` to the chosen `BASELINE_FOLDER` so it is also saved there. 

4. **Getting the true graph weights**: Now that we have all the necessary files extracted and organized, we can move on to the next phase, which involves calculating the true graph weights. We are doing this because we are using an approach inspired by [this paper](https://ftp.cs.ucla.edu/pub/stat_ser/r432.pdf). The `get_graph_weights.py` file handles this. Inside the file you need to specify the `BASELINE_FOLDER`. The true ATE details will be saved in a `true_edge_weights.json` file inside the `BASELINE_FOLDER` folder.
> Note: we do not need to do 4 anymore.


5. **Calculating ATE**: This is where the RMSE of true ATE and predicted ATE from the posterior samples are calculated. Given a baseline model and its set of posterior samples, let me walk you through the ATE calculation using one posterior sample -- which is essentially one predicted causal graph. Given a treatment and effect variables - `T` and `E` respectively -- we are interested in `ATE(T,E)`. I explain it in some detail [here](https://www.notion.so/chrisemezue/My-Mila-Project-29df7ef1d7954505abae8ab5361b2410?pvs=4#4e2ca9d22807470c80679d726652a679).

    For ATE calculation in general, the `run.sh` file handles its sbatching and looping through the baselines and seeds (or seed ranges).

    We currently have two approaches to this, distinguished by how we choose `T` and `E`:



    **👌🏽A: Total ATE**: Here we loop through all the possible treatment and effect variable combinations in our setting. Here we have 20 variables, so that makes it 400. This is obviously very computationally expensive, and we take some steps to improve on it. 

    > OLD: The file called `causal_inference.py` handles this type of ATE calculation. You need to specify some variables in the file: the baseline folder, where to save the predicted estimates, etc. Then tweak `run.sh` to run the required python file before finally doing `bash run.sh`.

    The file called `causal_inference_main.py` handles this type of ATE calculation. You need to specify some variables in the file: the baseline folder (`BASELINE_FOLDER`), where to save the predicted estimates (`ATE_DATAFRAME_FOLDER`), etc. 
    
    **UPDATE:** `job_main.sh` handles the job file for the parallelization of this. So run `sbatch job_main.sh` to initiate the whole process.    



    **👋🏽 B: ATE for true graphs in MEC**: For a more qualitative evaluation, we are not comparing against one true DAG but all the DAGs in its MEC. More details can be found [here](https://www.notion.so/chrisemezue/Evaluation-Details-7807d7cbf104474c95ca8e36cb3c507f).

    This calculation is very similar to **A** above. We have the file `causal_inference_true.py` which handles this. Like above, we need to specify some variables, specifically the baseline folder (`BASELINE_FOLDER`). The program saves the ATE csvs in a `variable_ates` folder within the `BASELINE_FOLDER` folder. To kick off this operation, run `sbatch job_true_main.sh`

    > Note: Before calculating the ATE for the true graphs, we need to get their CPDAG and then get their DAG samples from MEC. Therefore we need to run `get_cpdag.sh` and then `get_all_orientations_dag.sh` for the particular baseline of interest.



    **👋🏽 C: Special-case ATE**: Instead of looping through all the combinations, we focus on a few interesting treatment-effect cases and only calculate ATE for such variables. Further explanation can be found [here](https://www.notion.so/chrisemezue/Timeline-and-Experiments-to-run-7c02b1fe955749bfaaeccaa27423de3b?pvs=4#13bbfe1c482d40c2b60a968318e0a0b9).

    The file called `causal_inference_special_cases.py` handles the ATE for the special cases. Again, you need to set some variables inside, then specify the file inside `job_ci.sh`, before finally running `bash run.sh` to set it in motion.

    > **Result of this operation:** depending on the baseline models you chose, the result you should expect from this operation are CSV files showing the ATE for each seed and baseline model. Due to parallelization, it is chunked into multiple CSV files all housed in one folder, which is specified inside the corresponding python files above. You can then use these CSV files containing the results to plot or whatever.



6. **Plotting the results**: Here we want to plot the RMSE-ATEs that we got from the previous step. For that we use `plot.py`. We just need to specify the folder where we saved the CSV files: `FOLDER`. After that, we can easily run `python plot.py`. This does not require any GPUs and is relatively fast.

    > Note: the `plot.py` file, as of right now, needs some editing, especially to specify where to save the plots.
