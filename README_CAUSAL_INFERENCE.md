## Benchmarking causal discovery methods for the downstream task of cause effect estimation (ATE)

- Working document is [here](https://www.notion.so/chrisemezue/My-Mila-Project-29df7ef1d7954505abae8ab5361b2410)


## What to do
There are two parts to this project: the causal discovery pipeline and the causal inference (ATE) pipeline

### 1️⃣Causal Discovery pipeline
This involves training the causal discovery baseline models. 
Some important variables here are
- `num_variables`
- `num_samples`

1. Need to cd into `/home/mila/c/chris.emezue/gflownet_sl`. This is the codebase for all causal discovery.
2. The python file `eval_baselines_lingauss.py` handles causal discovery for `gadget`, `mc3`, `dibs`, and `bcdnets`.
3. Therefore to do causal discovery for the above baselines, simply run: `bash run_baselines_lingauss20.sh`.

    Inside `run_baselines_lingauss20.sh` you specify the number of variables, edges as well as  the `folder` -- this is an important parameter because it is where the experimental graph and data as well as the learned posterior will be stored. You will need this for phase 2️⃣.

4. **Running for `bootstrap_pc`, `bootstrap_ges`**: For these two bootstrap-based baselines, we have separated sh files for them -- `run_lingauss_bootstrap_pc.sh` for `bootstrap_pc`  and `run_lingauss_bootstrap_ges.sh` for `bootstrap_ges`. Again, here you can control the number of variables, edges as well as  the `folder`
5. **Running for dag-gfn**: The bash file called [`job.sh`](https://github.com/chrisemezue/gflownet_sl/blob/chris/ci/job.sh) handles it. Under the hood, it uses the `main.py` python file and stores all required files on WANDB. Running `job.sh` already enables parallelization by running the different seeds on separate cluster nodes.

    For WANDB, I use this tag structure `tags=[f'causal_inference_{args.num_samples}'] to distinguish the different run results I am running for the causal inference. Another thing I did was to configure the name of the wandb instance to be the seed number. All these were done to enable easy downloading of the required files for causal inference in phase 2️⃣.
