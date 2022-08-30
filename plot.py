# Continuous Embedding Meta Reinforcement Learning (CEMRL)

import os
import numpy as np
import click
import json
import torch
import torch.nn as nn


from configs.analysis_config import analysis_config
import matplotlib.ticker as ticker


from analysis.encoding import plot_encodings, plot_encodings_split
from analysis.progress_logger import plot
import matplotlib.pyplot as plt

def experiment(variant):
    # optional GPU mode

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    PLOT = variant['util_params']['plot']
    os.environ['DEBUG'] = str(int(DEBUG))
    os.environ['PLOT'] = str(int(PLOT))

    # create temp folder
    if not os.path.exists(variant['reconstruction_params']['temp_folder']):
        os.makedirs(variant['reconstruction_params']['temp_folder'])

    # showcase learned policy loaded
    showcase_itr = variant['showcase_itr']
    example_case = variant['analysis_params']['example_case']
    save = variant['analysis_params']['save']

    path_to_folder = variant['path_to_weights']
    path_to_folder_other = variant['path_to_weights_other']
    path_to_folder_other1 = variant['path_to_weights_other1']
    path_to_folder_other2 = variant['path_to_weights_other2']
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    # log file and plot progress file
    
    plot(path_to_folder,"CEMRL", color=3, save=save)
    plot(path_to_folder_other, "DMRLAD", color=0, save=save)
    #plot(path_to_folder_other1, "CEMRL+HFR", color=2, save=save)
    #plot(path_to_folder_other2, "CEMRL+DHFR", color=3, save=save)

    plt.grid()
    fontsize=14
    plt.legend(fontsize=12, loc='lower right')
    plt.xlabel("Training transition $n$", fontsize=fontsize)
    plt.ylabel("Average Return $R$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.option('--gpu', default=0)
@click.option('--num_workers', default=8)
@click.option('--use_mp', is_flag=True, default=False)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(gpu, use_mp, num_workers, docker, debug):

    variant = analysis_config

    path_to_folder = variant['path_to_weights']
    with open(os.path.join(os.path.join(path_to_folder, 'variant.json'))) as f:
        exp_params = json.load(f)
    variant["env_name"] = exp_params["env_name"]
    variant["env_params"] = deep_update_dict(exp_params["env_params"], variant["env_params"])
    variant["algo_params"] = deep_update_dict(exp_params["algo_params"], variant["algo_params"])
    variant["reconstruction_params"] = deep_update_dict(exp_params["reconstruction_params"], variant["reconstruction_params"])

    variant['util_params']['gpu_id'] = gpu
    variant['util_params']['use_multiprocessing'] = use_mp
    variant['util_params']['num_workers'] = num_workers

    # set other time steps than while training
    if variant["analysis_params"]["manipulate_time_steps"]:
        variant["algo_params"]["time_steps"] = variant["analysis_params"]["time_steps"]

    # set other time steps than while training
    if variant["analysis_params"]["manipulate_change_trigger"]:
        variant["env_params"] = deep_update_dict(variant["analysis_params"]["change_params"], variant["env_params"])

    # set other episode length than while training
    if variant["analysis_params"]["manipulate_max_path_length"]:
        variant["algo_params"]["max_path_length"] = variant["analysis_params"]["max_path_length"]

    # set other task number than while training
    if variant["analysis_params"]["manipulate_test_task_number"]:
        variant["env_params"]["n_eval_tasks"] = variant["analysis_params"]["test_task_number"]

    experiment(variant)


if __name__ == "__main__":
    main()
