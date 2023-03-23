## Benchmarking causal discovery methods for the downstream task of cause effect estimation (ATE)

- Working document is [here](https://www.notion.so/chrisemezue/My-Mila-Project-29df7ef1d7954505abae8ab5361b2410)


## What to do
There are two parts to this project: the causal discovery pipeline and the causal inference (ATE) pipeline

### 1️⃣Causal Discovery pipeline
This involves training the causal discovery baseline models. 


1. Need to cd into `/home/mila/c/chris.emezue/gflownet_sl`. This is the codebase for all causal discovery.
2. the python file `eval_baselines_lingauss.py` handles causal discovery for `gadget`, `mc3`, `dibs`, and `bcdnets`.
3. Therefore to do causal discovery for the above baselines, simply run 
```bash
bash run_baselines_lingauss20.sh
```
Inside `run_baselines_lingauss20.sh` you specify the number of variables, edges as well as  the `folder` -- this is an important parameter because it is where the experimental graph and data as well as the learned posterior will be stored. You will need this for phase 2️⃣.
4. **Running for `bootstrap_pc`, `bootstrap_ges`**: For these two bootstrap-based baselines, we have separated sh files for them -- `run_lingauss_bootstrap_pc.sh` for `bootstrap_pc`  and `run_lingauss_bootstrap_ges.sh` for `bootstrap_ges`. Again, here you can control the number of variables, edges as well as  the `folder`
