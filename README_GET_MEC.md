# Getting the MEC for the truth graph

Here describes our work to get the DAG samples in the MEC of the truth graph.

1. First thing we do is to get the CPDAG given the truth graph. For that we have `get_cpdag.sh`. Under the hood, it uses the `get_cpdag.py` file. You specify the `BASELINE_FOLDER` variable. Then goes into all the baselines and all the seeds in that folder. Then it creates the cpdag for each graph, and saves it as a pickle (using the filename `true_cpdag.pkl`) in the same destination of the truth graph.
> This does not require sbatch. It can be easily done with `bash get_cpdag.sh` because it does not take time.

2. Now that we have the CPDAG, we get the MEC samples, by taking all the DAG samples from the CPDAG. To handle this we use the `get_all_orientations_dag.sh`. Under the hood, it uses `get_all_orientations_dag.py` file. You specify the `BASELINE_FOLDER` variable. Then goes into all the baselines and all the seeds in that folder. Then it opens the corresponding `true_cpdag.pkl`, gets all the DAG samples and saves them as a pickled array called `true_mec_dags.pkl`.


# Getting the ATE for the MEC samples

