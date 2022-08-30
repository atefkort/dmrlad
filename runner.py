# Distributional Meta-RL with Augmented Data (DMRLAD)

import os
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
from configs.default import default_config


from dmrlad.encoder_decoder_networks import PriorPz, EncoderMixtureModelTrajectory, EncoderMixtureModelTransitionSharedY, EncoderMixtureModelTransitionIndividualY, DecoderMDP
from dmrlad.dsac import PolicyTrainer
from dmrlad.stacked_replay_buffer import StackedReplayBuffer
from dmrlad.reconstruction_trainer import ReconstructionTrainer
from dmrlad.rollout_worker import RolloutCoordinator
from dmrlad.agent import CEMRLAgent, ScriptedPolicyAgent
from dmrlad.relabeler import Relabeler
from dmrlad.dmrlad_algorithm import DMRLADAlgorithm


def experiment(variant):
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    torch.set_num_threads(1)
    if(variant['algo_params']['use_fixed_seeding']):
        torch.manual_seed(variant['algo_params']['seed'])
        np.random.seed(variant['algo_params']['seed'])

    # create logging directory
    encoding_save_epochs = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450,
                            475, 500, 600, 750, 1000, 1250, 1500, 1750, 2000, 3000, 5000, 10000, 15000, 20000]
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=None,
                                      base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='specific',
                                      snapshot_gap=variant['algo_params']['snapshot_gap'],
                                      snapshot_points=encoding_save_epochs)

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
    agent_class = ScriptedPolicyAgent if variant['env_params']['scripted_policy'] else CEMRLAgent
    agent = agent_class(
        encoder,
        prior_pz,
        policy
    )
    
    policy_trainer = PolicyTrainer(
        policy,
        target_policy,
        zf1,
        zf2,
        target_zf1,
        target_zf2,
        alpha_net,
        replay_buffer,
        variant['algo_params']['batch_size_policy'],
        action_dim,
        variant['algo_params']['data_usage_sac'],
        use_parametrized_alpha=variant['algo_params']['use_parametrized_alpha'],
        target_entropy_factor=variant['algo_params']['target_entropy_factor'],
        alpha=variant['algo_params']['sac_alpha'],
        fp=fp,
        target_fp=target_fp,
        num_quantiles=num_quantiles,
    )

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

    # ReconstructionTrainer
    reconstruction_trainer = ReconstructionTrainer(
        encoder,
        decoder,
        prior_pz,
        replay_buffer,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        latent_dim,
        time_steps,
        variant['reconstruction_params']['lr_decoder'],
        variant['reconstruction_params']['lr_encoder'],
        variant['reconstruction_params']['alpha_kl_z'],
        variant['reconstruction_params']['beta_kl_y'],
        variant['reconstruction_params']['use_state_diff'],
        variant['reconstruction_params']['component_constraint_learning'],
        variant['env_params']['state_reconstruction_clip'],
        variant['reconstruction_params']['train_val_percent'],
        variant['reconstruction_params']['eval_interval'],
        variant['reconstruction_params']['early_stopping_threshold'],
        experiment_log_dir,
        variant['reconstruction_params']['prior_mode'],
        variant['reconstruction_params']['prior_sigma'],
        True if variant['algo_params']['encoding_mode'] == 'transitionIndividualY' else False,
        variant['algo_params']['data_usage_reconstruction'],
    )


    # PolicyTrainer



    relabeler = Relabeler(
        encoder,
        replay_buffer,
        variant['algo_params']['batch_size_relabel'],
        action_dim,
        obs_dim,
        variant['algo_params']['use_data_normalization'],
    )


    algorithm = DMRLADAlgorithm(
        replay_buffer,
        rollout_coordinator,
        reconstruction_trainer,
        policy_trainer,
        relabeler,
        agent,
        networks,

        train_tasks,
        test_tasks,
        train_goals,
        
        variant['algo_params']['utility_batch_size'],
        variant['algo_params']['num_train_epochs'],
        variant['algo_params']['num_reconstruction_steps'],
        variant['algo_params']['num_policy_steps'],
        variant['algo_params']['num_train_tasks_per_episode'],
        variant['algo_params']['num_transitions_initial'],
        variant['algo_params']['num_transitions_per_episode'],
        variant['algo_params']['num_eval_trajectories'],
        variant['algo_params']['showcase_every'],
        variant['algo_params']['snapshot_gap'],
        variant['algo_params']['num_showcase_deterministic'],
        variant['algo_params']['num_showcase_non_deterministic'],
        variant['algo_params']['use_relabeler'],
        experiment_log_dir,
        latent_dim
        )

    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    PLOT = variant['util_params']['plot']
    os.environ['DEBUG'] = str(int(DEBUG))
    os.environ['PLOT'] = str(int(PLOT))

    # create temp folder
    if not os.path.exists(variant['reconstruction_params']['temp_folder']):
        os.makedirs(variant['reconstruction_params']['temp_folder'])

    # run the algorithm
    algorithm.train()


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
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--num_workers', default=4)
@click.option('--use_mp', is_flag=True, default=False)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, use_mp, num_workers, docker, debug):

    variant = default_config

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['util_params']['use_multiprocessing'] = use_mp
    variant['util_params']['num_workers'] = num_workers


    experiment(variant)

if __name__ == "__main__":
    main()
