import os
import subprocess



#https://stackoverflow.com/a/89243

ROOT_FOLDER= '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20'
print('='*30+f' WORKING ON: {ROOT_FOLDER} '+'='*30)
for baseline in ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3','dag-gfn']:
    FOLDER = os.path.join(ROOT_FOLDER,baseline)
    #SEEDS = [os.path.join(FOLDER,f.name) for f in os.scandir(FOLDER)]
    SEEDS = [os.path.join(FOLDER,str(i)) for i in range(26)]
    undone = 0

    for seed in SEEDS:
        cmd_str = f"ls {seed}/variable_ates/ | wc -l"
        #subprocess.run(cmd_str, shell=True)
        return_code = subprocess.check_output(cmd_str, shell=True)
        #breakpoint()
        if return_code.decode().strip()!='380':
            undone+=1
            #print(f'Undone for: {seed_path}')


    print(f'{undone} true ATE experiments are not done for {baseline}.')
