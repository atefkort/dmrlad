# Continuous Embedding Meta Reinforcement Learning (CEMRL)

import os, shutil
import pathlib
import numpy as np
import click
import json
import torch
import torch.nn as nn

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import Mlp, FlattenMlp, QuantileMlp, softmax
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.analysis_config import analysis_config


from dmrlad.encoder_decoder_networks import PriorPz, EncoderMixtureModelTrajectory, EncoderMixtureModelTransitionSharedY, EncoderMixtureModelTransitionIndividualY, DecoderMDP
from dmrlad.stacked_replay_buffer import StackedReplayBuffer
from dmrlad.rollout_worker import RolloutCoordinator
from dmrlad.agent import CEMRLAgent

import pickle

from analysis.encoding import plot_encodings, plot_encodings_split
from analysis.progress_logger import manage_logging


def experiment(variant):
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    torch.set_num_threads(1)
    if variant['algo_params']['use_fixed_seeding']:
        torch.manual_seed(variant['algo_params']['seed'])
        np.random.seed(variant['algo_params']['seed'])



    # create multi-task environment and sample tasks
    env = ENVS[variant['env_name']](**variant['env_params'])
    if variant['env_params']['use_normalized_env']:
        env = NormalizedBoxEnv(env)
    if variant['train_or_showcase'] == 'showcase':
        env = CameraWrapper(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    tasks = list(range(len(env.tasks)))
    #train_tasks = tasks[:variant['env_params']['n_train_tasks']]
    train_tasks = list(range(len(env.train_tasks)))
    test_tasks = tasks[-variant['env_params']['n_eval_tasks']:]
    train_goals = env.get_train_goals()

    # instantiate networks
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']

    # encoder used: single transitions or trajectories
    if variant['algo_params']['encoding_mode'] == 'transitionSharedY':
        encoder_input_dim = obs_dim + action_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelTransitionSharedY
    elif variant['algo_params']['encoding_mode'] == 'transitionIndividualY':
        encoder_input_dim = obs_dim + action_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelTransitionIndividualY
    elif variant['algo_params']['encoding_mode'] == 'trajectory':
        encoder_input_dim = time_steps * (obs_dim + action_dim + reward_dim + obs_dim)
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelTrajectory
    else:
        raise NotImplementedError

    encoder = encoder_model(
        shared_dim,
        encoder_input_dim,
        latent_dim,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        time_steps,
        variant['algo_params']['merge_mode']
    )

    decoder = DecoderMDP(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
    )

    prior_pz = PriorPz(num_classes, latent_dim)

    M = variant['algo_params']['sac_layer_size']
    num_quantiles = variant['algo_params']['num_quantiles']

    zf1 = QuantileMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M],
    )
    zf2 = QuantileMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M],
    )
    target_zf1 = QuantileMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M],
    )
    target_zf2 = QuantileMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M],
    )

    policy = TanhGaussianPolicy(
        obs_dim=(obs_dim + latent_dim),
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_sizes=[M, M, M],
    )

    target_policy = TanhGaussianPolicy(
        obs_dim=(obs_dim + latent_dim),
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_sizes=[M, M, M],
    )

    fp = target_fp = None
    if variant['algo_params'].get('tau_type') == 'fqf':
        fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
        target_fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )

    alpha_net = Mlp(
        hidden_sizes=[latent_dim * 10],
        input_size=latent_dim,
        output_size=1
    )

    networks = {'encoder': encoder,
                'prior_pz': prior_pz,
                'decoder': decoder,
                'zf1': zf1,
                'zf2': zf2,
                'target_zf1': target_zf1,
                'target_zf2': target_zf2,
                'policy': policy,
                'target_policy':target_policy,
                'alpha_net': alpha_net}

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        itr = variant['showcase_itr']
        path = variant['path_to_weights']
        for name, net in networks.items():
            net.load_state_dict(torch.load(os.path.join(path, name + '_itr_' + str(itr) + '.pth'), map_location='cpu'))

    replay_buffer = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        obs_dim,
        action_dim,
        latent_dim,
        variant['algo_params']['data_usage_reconstruction'],
        variant['algo_params']['data_usage_sac'],
        variant['algo_params']['num_last_samples'],
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        train_tasks
    )

    #Agent
    agent = CEMRLAgent(
        encoder,
        prior_pz,
        policy,
    )

    if ptu.gpu_enabled():
        agent.to(ptu.device)

    # Rollout Coordinator
    rollout_coordinator = RolloutCoordinator(
        env,
        variant['env_name'],
        variant['env_params'],
        variant['train_or_showcase'],
        agent,
        zf1,
        zf2,
        num_quantiles,
        replay_buffer,
        time_steps,
        train_goals,
        variant['algo_params']['utility_batch_size'],
        variant['algo_params']['sparse_rewards'],
        variant['algo_params']['use_softmax'],

        variant['algo_params']['max_path_length'],
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        variant['util_params']['use_multiprocessing'],
        variant['algo_params']['use_data_normalization'],
        variant['util_params']['num_workers'],
        variant['util_params']['gpu_id'],
        variant['env_params']['scripted_policy']
        )


    



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
    replay_buffer.stats_dict = pickle.load(open(os.path.join(path_to_folder, "replay_buffer_stats_dict_" + str(showcase_itr) + ".p"), "rb"))
    env.reset_task(np.random.randint(len(env.test_tasks)) + len(env.train_tasks))
    env.set_meta_mode('test')

    # log file and plot progress file
    if variant['analysis_params']['log_and_plot_progress']:
        manage_logging(path_to_folder, save=save)

    # plot encodings
    if variant['analysis_params']['plot_encoding']:
        plot_encodings_split(showcase_itr, path_to_folder, save=save)


    test = True
    if variant['analysis_params']['plot_test_tasks']:
        import matplotlib.pyplot as plt
        env_tasks = [task["velocity"] for task in env.tasks]
        
        results = rollout_coordinator.collect_data(test_tasks, 'test',
                deterministic=False, max_trajs=1, animated=variant['analysis_params']['visualize_run'], save_frames=False)
        per_path_rewards = [np.sum(path["rewards"]) for worker in results for task in worker for path in task[0]]
        per_path_rewards = np.array(per_path_rewards)
        cur_tasks = [env_tasks[path["task_id"]] for worker in results for task in worker for path in task[0]]
        cur_tasks = np.array(cur_tasks)
        plt.plot(cur_tasks, per_path_rewards, 'b*', label='sample')
        plt.xlabel('tasks')
        plt.ylabel('rewards')
        plt.legend()
        plt.show()    
    
    temp_tasks = [test_tasks[1],test_tasks[4],test_tasks[7],test_tasks[8],test_tasks[5]]   
    # velocity plot
    if 'vel' in variant['env_name'].split('-')  and variant['analysis_params']['plot_time_response']:
        import matplotlib.pyplot as plt
        plt.figure()
        results = rollout_coordinator.collect_data(test_tasks[example_case:example_case + 1], 'test', deterministic=True, max_trajs=1, animated=False, save_frames=True)
        velocity_is = [a['velocity'] for a in results[0][0][0][0]['env_infos']]
        filter_constant = time_steps
        velocity_is_temp = ([0] * filter_constant) + velocity_is
        velocity_is_filtered = []
        for i in range(len(velocity_is)):
            velocity_is_filtered.append(sum(velocity_is_temp[i:i+filter_constant]) / filter_constant)
        velocity_goal = [a['true_task']['specification'] for a in results[0][0][0][0]['env_infos']]
        plt.plot(list(range(len(velocity_goal))), velocity_goal, label="goal velocity")
        plt.plot(list(range(len(velocity_is))), velocity_is, label="velocity")
        plt.plot(list(range(len(velocity_is_filtered))), velocity_is_filtered, label="velocity filtered")
        plt.xlabel("time $t$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "velocity_vs_goal_velocity_new" + ".pdf", dpi=300, format="pdf")
        plt.show()
    if variant['env_name'].split('-')[-1] == 'dir' and variant['analysis_params']['plot_time_response']:
        import matplotlib.pyplot as plt
        figsize=None
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes_tuple = plt.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw={'height_ratios': [1, 1]}, figsize=figsize)
        direction_is = [a['direction'] for a in results[0][0][0][0]['env_infos']]
        direction_goal = [a['true_task']['specification'] for a in results[0][0][0][0]['env_infos']]
        axes_tuple[0].plot(list(range(len(direction_is))), direction_is, color=cycle[0], label="velocity")
        #axes_tuple[1].plot(list(range(len(direction_goal))), np.sign(direction_is), color=cycle[1], label="direction")
        axes_tuple[1].plot(list(range(len(direction_goal))), direction_goal, color=cycle[1], label="goal direction")
        axes_tuple[0].grid()
        #axes_tuple[1].grid()
        axes_tuple[1].grid()
        axes_tuple[0].legend(loc='upper right')
        axes_tuple[1].legend(loc='upper right')
        #axes_tuple[2].legend(loc='lower left')
        axes_tuple[1].set_xlabel("time $t$")
        plt.tight_layout()
        if save:
            plt.savefig(path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "velocity_vs_goal_direction_new" + ".pdf", dpi=300, bbox_inches='tight', format="pdf")
        plt.show()

    if 'vel' in variant['env_name'].split('-')  and variant['analysis_params']['plot_velocity_multi']:
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pl
        plt.figure(figsize=(10,5))
        colors = pl.cm.coolwarm(np.linspace(0, 1, len(temp_tasks)))
        #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(temp_tasks)):
            results = rollout_coordinator.collect_data(temp_tasks[i:i + 1], 'test', deterministic=False, max_trajs=1,
                                                       animated=False, save_frames=False)

            velocity_is = [a['velocity'] for a in results[0][0][0][0]['env_infos']]
            velocity_goal = [a['true_task']['specification'] for a in results[0][0][0][0]['env_infos']]
            plt.plot(list(range(len(velocity_goal))), velocity_goal, '--', color=colors[i])
            plt.plot(list(range(len(velocity_is))), velocity_is, color=colors[i])

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='gray', linestyle='--'),
                        Line2D([0], [0], color='gray')]

        fontsize = 14
        plt.legend(custom_lines, ['goal velocity', 'velocity'], fontsize=fontsize, loc='lower right')
        plt.xlabel("time step $t$", fontsize=fontsize)
        plt.ylabel("velocity $v$", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.xlim(0, len(list(range(len(velocity_goal)))))
        #plt.title("cheetah-stationary-vel: velocity vs. goal velocity", fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "multiple_velocity_vs_goal_velocity" + ".pdf", dpi=300, format="pdf")
        plt.show()
    # video taking
    if variant['analysis_params']['produce_video']:
        print("Producing video... do NOT kill program until completion!")
        video_name_string = path_to_folder.split('/')[-1] + "_" + variant['env_name'] + ".mp4"
        results = rollout_coordinator.collect_data(test_tasks[example_case:example_case + 1], 'test', deterministic=False, max_trajs=1, animated=False, save_frames=True)
        env_test_tasks = [task["goal"] for task in env.test_tasks]
        print("Goal to reach: " + str(env_test_tasks[example_case]))
        vel = [info["xpos"] for worker in results for task in worker for path in task[0] for info in path["env_infos"]]
        per_path_rewards = np.array(vel)
        eval_average_reward = np.median(per_path_rewards)
        #print(per_path_rewards)
        print("Median vel: " + str(eval_average_reward))
        per_path_rewards = [np.sum(path["rewards"]) for worker in results for task in worker for path in task[0]]
        per_path_rewards = np.array(per_path_rewards)
        eval_average_reward = per_path_rewards.mean()
        print("Average reward: " + str(eval_average_reward))
        path_video = results[0][0][0][0]
        video_frames = []
        video_frames += [t['frame'] for t in path_video['env_infos']]
        print("Saving video...")
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))
        video_filename = os.path.join(path_to_folder, video_name_string)
        # run ffmpeg to make the video
        os.system('ffmpeg -r 25 -i {}/%06d.jpg -vb 20M -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)


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
