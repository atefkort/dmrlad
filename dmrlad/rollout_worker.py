import numpy as np
import torch
import ray
import os

from scipy.special import softmax
from collections import OrderedDict

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
import rlkit.torch.pytorch_util as ptu


class RolloutCoordinator:
    def __init__(self,
                 env,
                 env_name,
                 env_args,
                 train_or_showcase,
                 agent,
                 zf1,
                 zf2,
                 num_quantiles,
                 replay_buffer,
                 time_steps,
                 train_goals,
                 utility_batch_size,
                 sparse_rewards,
                 use_softmax,

                 max_path_length,
                 permute_samples,
                 encoding_mode,

                 use_multiprocessing,
                 use_data_normalization,
                 num_workers,
                 gpu_id,
                 scripted_policy
                 ):
        self.env = env
        self.env_name = env_name
        self.env_args = env_args
        self.train_or_showcase = train_or_showcase
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.time_steps = time_steps
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples
        self.encoding_mode = encoding_mode

        self.use_multiprocessing = use_multiprocessing
        self.use_data_normalization = use_data_normalization
        self.num_workers = num_workers
        self.gpu_id = gpu_id
        self.scripted_policy = scripted_policy

        self.num_env_steps = 0
        self.train_goals = train_goals
        self.n_train_goals = len(train_goals)
        self.utility_batch_size = utility_batch_size
        self.sparse_rewards = sparse_rewards
        self.use_softmax = use_softmax
        self.zf1 = zf1
        self.zf2 = zf2
        self.num_quantiles = num_quantiles
        
        if self.use_multiprocessing:
            ray.init(
                #log_to_driver=False
                # memory=1000 * 1024 * 1024,
                # object_store_memory=2500 * 1024 * 1024,
                # driver_object_store_memory=1000 * 1024 * 1024
            )

    def collect_data(self, tasks, train_test, deterministic=False, max_samples=np.inf, max_trajs=np.inf, animated=False, save_frames=False):
        # distribute tasks over workers
        tasks_per_worker = [[] for _ in range(self.num_workers)]
        counter = 0
        for task in tasks:
            if counter % self.num_workers == 0:
                counter = 0
            tasks_per_worker[counter].append(task)
            counter += 1

        if self.use_multiprocessing:
            # put on cpu before starting ray
            self.agent.to('cpu')
            self.agent.policy.to('cpu')
            self.agent.encoder.to('cpu')
            workers = [RemoteRolloutWorker.remote(None, self.env_name, self.env_args, self.train_or_showcase,
                                                  self.agent, self.time_steps, self.max_path_length, self.permute_samples, self.encoding_mode, self.gpu_id, self.scripted_policy,
                                                  self.use_multiprocessing, self.use_data_normalization, self.replay_buffer.stats_dict,
                                                  task_list, self.env.tasks, self.env.train_tasks, self.env.test_tasks) for task_list in tasks_per_worker]
            results = ray.get([worker.obtain_samples_from_list.remote(train_test,
                deterministic=deterministic, max_samples=max_samples, max_trajs=max_trajs, animated=animated,
                save_frames=save_frames) for worker in workers])
        else:
            workers = [RolloutWorker(self.env, self.env_name, self.env_args, self.train_or_showcase,
                                     self.agent, self.time_steps, self.max_path_length, self.permute_samples, self.encoding_mode, self.gpu_id, self.scripted_policy,
                                     self.use_multiprocessing, self.use_data_normalization, self.replay_buffer.stats_dict,
                                     task_list, self.env.tasks, self.env.train_tasks, self.env.test_tasks) for task_list in tasks_per_worker]
            results = [[worker.obtain_samples(task, train_test,
                deterministic=deterministic, max_samples=max_samples, max_trajs=max_trajs, animated=animated,
                save_frames=save_frames
            ) for task in task_list] for worker, task_list in zip(workers, tasks_per_worker)]

        self.agent.to(ptu.device)
        self.agent.policy.to(ptu.device)
        self.agent.encoder.to(ptu.device)
        return results
    
    def collect_initial_replay_data(self, tasks, max_samples=np.inf):
        num_env_steps = 0
        results = self.collect_data(tasks, 'train', deterministic=False, max_samples=max_samples, animated=False)
        for worker in results:
            for task in worker:
                for path in task[0]:
                    self.replay_buffer.add_episode(path)
                num_env_steps += task[1]
        return num_env_steps
    
    def collect_replay_data(self, tasks, log_Z, max_samples=np.inf):
        num_env_steps = 0
        results = self.collect_data(tasks, 'train', deterministic=False, max_samples=max_samples, animated=False)
        
        print("Relabeling..." )
        for worker in results:
            for task in worker:
                for path in task[0]:
                    self.replay_buffer.add_episode(path)
                    self.relabel_dhfr(path, log_Z)
                num_env_steps += task[1]
        return num_env_steps

    def evaluate(self, train_test, tasks, num_eval_trajectories, deterministic=True, animated=False, save_frames=False, log=True):
        results = self.collect_data(tasks, 'train_test', deterministic=deterministic, max_trajs=num_eval_trajectories, animated=animated, save_frames=save_frames)
        eval_statistics = OrderedDict()
        if log:
            deterministic_string = '_deterministic' if deterministic else '_non_deterministic'
            per_path_rewards = [np.sum(path["rewards"]) for worker in results for task in worker for path in task[0]]
            per_path_rewards = np.array(per_path_rewards)
            eval_average_reward = per_path_rewards.mean()
            eval_std_reward = per_path_rewards.std()
            eval_max_reward = per_path_rewards.max()
            eval_min_reward = per_path_rewards.min()
            eval_statistics[train_test + '_eval_avg_reward' + deterministic_string] = eval_average_reward
            eval_statistics[train_test + '_eval_std_reward' + deterministic_string] = eval_std_reward
            eval_statistics[train_test + '_eval_max_reward' + deterministic_string] = eval_max_reward
            eval_statistics[train_test + '_eval_min_reward' + deterministic_string] = eval_min_reward
            # success rates for meta world
            if "success" in results[0][0][0][0]["env_infos"][0]:
                success_values = np.array([sum([timestep["success"] for timestep in path["env_infos"]]) for worker in results for task in worker for path in task[0]])
                success_rate = np.sum((success_values > 0).astype(int)) / success_values.shape[0]
                eval_statistics[train_test + '_eval_success_rate'] = success_rate
            if int(os.environ['DEBUG']) == 1:
                print(train_test + ":")
                print("Mean reward: " + str(eval_average_reward))
                print("Std reward: " + str(eval_std_reward))
                print("Max reward: " + str(eval_max_reward))
                print("Min reward: " + str(eval_min_reward))
            return eval_average_reward, eval_std_reward, eval_max_reward, eval_min_reward, eval_statistics
        else:
            return

    def relabel_dhfr(
        self,
        path,
        log_Z
    ):
        data = dict(
            observations = path['observations'],
            actions = path['actions'],
            rewards = path['rewards'],
            next_observations = path['next_observations'],
        )
        
        q_vals = np.zeros((self.n_train_goals))
        log_pis = np.zeros((self.n_train_goals))

        for idx, goal in enumerate(self.train_goals):
            if self.sparse_rewards:
                rewards = np.array(
                    [self.env.reward(info, goal)[1] for info in path["env_infos"]]
                )
            else:
                rewards = np.array(
                    [self.env.reward(info, goal)[0] for info in path["env_infos"]]
                )
            data["rewards"] = rewards

            start_obs, _ = self.replay_buffer.sample_start_obs([idx], batch_size=self.utility_batch_size)                
            task_z = self.infer_context(data, start_obs[0])
            start_obs = ptu.from_numpy(start_obs)
            t, b, _ = start_obs.size()
            start_obs = start_obs.view(t * b, -1)            
            task_z = ptu.from_numpy(task_z)
            
            obs = torch.cat((start_obs, task_z), dim=1)
            with torch.no_grad():
    
                actions, policy_mean, policy_log_std, log_pi, *_ = self.agent.policy(
                obs, reparameterize=True, return_log_prob=True,
                )

                next_tau, next_tau_hat, next_presum_tau = self.get_tau(actions)
                target_z1_values = self.zf1(obs, actions, next_tau_hat)
                target_z2_values = self.zf2(obs, actions, next_tau_hat)
                min_qs = torch.min(target_z1_values, target_z2_values)
            q_vals[idx] = ptu.get_numpy(torch.mean(min_qs))
            log_pis[idx] = ptu.get_numpy(torch.mean(log_pi))
        
        dist = softmax(q_vals - log_Z)
        if self.use_softmax:
            task_idx = np.random.choice(self.n_train_goals, p=dist)
        else:
            task_idx = np.argmax(dist)
        rewards_and_done = [
            self.env.reward(info, self.train_goals[task_idx])
            for info in path["env_infos"]
            ]
        
        if self.sparse_rewards:
            rewards = np.array([rd[0] for rd in rewards_and_done])
            sparse_rewards = [rd[1] for rd in rewards_and_done]
            terminals = np.array([rd[2] for rd in rewards_and_done])
            for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                info.update({"sparse_reward": sparse_reward})
        else:
            rewards = np.array([rd[0] for rd in rewards_and_done])
            terminals = np.array([rd[1] for rd in rewards_and_done])
        
        path["rewards"] = rewards.reshape(-1, 1)
        path["terminals"] = terminals.reshape(-1, 1)   
        self.replay_buffer.add_episode(path)

    def infer_context(self, data, start_obs):
        self.action_space = self.env.action_space.low.size
        self.obs_space = self.env.observation_space.low.size
        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        next_observations = data['next_observations']
        task_indicators = []
        self.context = torch.zeros((self.time_steps, self.obs_space + self.action_space + 1 + self.obs_space))
        action_space = int(np.prod(self.env.action_space.shape))
        for o,a,r,next_o in zip(observations, actions, rewards, next_observations):
            agent_input = self.build_encoder_input(o, self.context, action_space)
            self.update_context(o, a, np.array([r], dtype=np.float32), next_o)
        
        for o in start_obs:   
            out = self.agent.get_action(agent_input, o, deterministic=False, z_debug=None, env=self.env)
            task_indicator = out[1]
            task_indicators.append(task_indicator)
        
        task_indicators = np.array(task_indicators)
        if len(observations.shape) == 1:
            task_indicators.np.expand_dim(task_indicators, 1)
        
        return task_indicators
    
    def update_context(self, o, a, r, next_o):
        if self.use_data_normalization and self.replay_buffer.stats_dict is not None:
            stats_dict = self.replay_buffer.stats_dict
            o = torch.from_numpy((o - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
            a = torch.from_numpy((a - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
            r = torch.from_numpy((r - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-9))
            next_o = torch.from_numpy((next_o - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-9))
        else:
            o = torch.from_numpy(o)
            a = torch.from_numpy(a)
            r = torch.from_numpy(r)
            next_o = torch.from_numpy(next_o)
        data = torch.cat([o, a, r, next_o]).view(1, -1)
        context = torch.cat([self.context, data], dim=0)
        context = context[-self.time_steps:]
        self.context = context

    def build_encoder_input(self, obs, context, action_space):
        encoder_input = context.detach().clone()

        if self.permute_samples:
            perm = torch.LongTensor(torch.randperm(encoder_input.shape[0]))
            encoder_input = encoder_input[perm]

        if self.encoding_mode == 'trajectory':
            encoder_input = encoder_input.view(1, -1)
        if self.encoding_mode == 'transitionSharedY' or self.encoding_mode == 'transitionIndividualY':
            encoder_input.unsqueeze_(0)
        return encoder_input.to(ptu.device)
    
    def get_tau(self, actions):
        
        presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)

        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

class RolloutWorker:
    def __init__(self,
                 env,
                 env_name,
                 env_args,
                 train_or_showcase,
                 agent,
                 time_steps,
                 max_path_length,
                 permute_samples,
                 encoding_mode,
                 gpu_id,
                 scripted_policy,
                 use_multiprocessing,
                 use_data_normalization,
                 replay_buffer_stats_dict,
                 task_list,
                 env_tasks,
                 env_train_tasks,
                 env_test_tasks
                 ):
        if use_multiprocessing:
            environment = ENVS[env_name](**env_args)
            if env_args['use_normalized_env']:
                environment = NormalizedBoxEnv(environment)
            if train_or_showcase == 'showcase':
                environment = CameraWrapper(environment)
            self.env = environment
        else:
            self.env = env
        self.agent = agent
        self.time_steps = time_steps
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples
        self.encoding_mode = encoding_mode
        self.gpu_id = gpu_id
        self.scripted_policy = scripted_policy
        self.use_data_normalization = use_data_normalization

        self.replay_buffer_stats_dict = replay_buffer_stats_dict
        self.task_list = task_list

        self.env.tasks = env_tasks
        self.env.train_tasks = env_train_tasks
        self.env.train_tasks = env_test_tasks

        self.action_space = self.env.action_space.low.size
        self.obs_space = self.env.observation_space.low.size
        self.context = None

    def obtain_samples_from_list(self, train_test, deterministic=False, max_samples=np.inf, max_trajs=np.inf, animated=False, save_frames=False):
        results = []
        for task in self.task_list:
            result = self.obtain_samples(task, train_test, deterministic=deterministic, max_samples=max_samples, max_trajs=max_trajs, animated=animated, save_frames=save_frames)
            results.append(result)

        return results

    def obtain_samples(self, task, train_test, deterministic=False, max_samples=np.inf, max_trajs=np.inf, animated=False, save_frames=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        """

        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            self.env.reset_task(task)
            self.env.set_meta_mode(train_test)
            path = self.rollout(deterministic=deterministic, max_path_length=self.max_path_length if max_samples - n_steps_total > self.max_path_length else max_samples - n_steps_total, animated=animated, save_frames=save_frames)
            path['task_id'] = task
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
        return paths, n_steps_total

    def rollout(self, deterministic=False, max_path_length=np.inf, animated=False, save_frames=False):

        observations = []
        task_indicators = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        self.context = torch.zeros((self.time_steps, self.obs_space + self.action_space + 1 + self.obs_space))
        action_space = int(np.prod(self.env.action_space.shape))

        if self.scripted_policy:
            self.env.metaworld_env._partially_observable = False

        o = self.env.reset()
        next_o = None
        path_length = 0

        #debug
        #true_task = torch.tensor([[1.0]])

        if animated:
            self.env.render()
        while path_length < max_path_length:
            agent_input = self.build_encoder_input(o, self.context, action_space)
            out = self.agent.get_action(agent_input, o, deterministic=deterministic, z_debug=None, env=self.env)
            a, agent_info = out[0]
            task_indicator = out[1]
            next_o, r, d, env_info = self.env.step(a)
            self.update_context(o, a, np.array([r], dtype=np.float32), next_o)
            if self.scripted_policy:
                observations.append(np.hstack((o[0:6], np.zeros(6))))
            else:
                observations.append(o)
            task_indicators.append(task_indicator)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            #debug
            #true_task = torch.tensor([[1.0]]) if env_info['true_task'] == 1 else torch.tensor([[-1.0]])
            path_length += 1
            o = next_o
            if animated:
                self.env.render()
            if save_frames:
                from PIL import Image
                image = Image.fromarray(np.flipud(self.env.get_image()))
                 # make even higher for better quality
                env_info['frame'] = image
            env_infos.append(env_info)
            if d:
                break

        agent_input = self.build_encoder_input(next_o, self.context, action_space)
        next_task_indicator = self.agent.get_action(agent_input, next_o, deterministic=deterministic, env=self.env)[1]
        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        task_indicators = np.array(task_indicators)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            task_indicators.np.expand_dim(task_indicators, 1)
            next_o = np.array([next_o])
            next_task_indicator = np.array([next_task_indicator])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        next_task_indicators = np.vstack(
            (
                task_indicators[1:, :],
                np.expand_dims(next_task_indicator, 0)
            )
        )

        true_tasks = [env_infos[i]['true_task'] for i in range(len(env_infos))]

        return dict(
            observations=observations,
            task_indicators=task_indicators,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            next_task_indicators=next_task_indicators,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            true_tasks=np.array(true_tasks).reshape(-1, 1),
        )

    def update_context(self, o, a, r, next_o):
        if self.use_data_normalization and self.replay_buffer_stats_dict is not None:
            stats_dict = self.replay_buffer_stats_dict
            o = torch.from_numpy((o - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
            a = torch.from_numpy((a - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
            r = torch.from_numpy((r - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-9))
            next_o = torch.from_numpy((next_o - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-9))
        else:
            o = torch.from_numpy(o)
            a = torch.from_numpy(a)
            r = torch.from_numpy(r)
            next_o = torch.from_numpy(next_o)
        data = torch.cat([o, a, r, next_o]).view(1, -1)
        context = torch.cat([self.context, data], dim=0)
        context = context[-self.time_steps:]
        self.context = context

    def build_encoder_input(self, obs, context, action_space):
        encoder_input = context.detach().clone()

        if self.permute_samples:
            perm = torch.LongTensor(torch.randperm(encoder_input.shape[0]))
            encoder_input = encoder_input[perm]

        if self.encoding_mode == 'trajectory':
            encoder_input = encoder_input.view(1, -1)
        if self.encoding_mode == 'transitionSharedY' or self.encoding_mode == 'transitionIndividualY':
            encoder_input.unsqueeze_(0)
        return encoder_input.to(ptu.device)


@ray.remote
class RemoteRolloutWorker(RolloutWorker):
    def __init__(self, env, env_name, env_args, train_or_showcase, agent, time_steps, max_path_length, permute_samples, encoding_mode, gpu_id,  scripted_policy, use_multiprocessing, use_data_normalization, replay_buffer_stats_dict, task_list, tasks, train_tasks, test_tasks):
        super().__init__(env, env_name, env_args, train_or_showcase, agent, time_steps, max_path_length, permute_samples, encoding_mode, gpu_id,  scripted_policy, use_multiprocessing, use_data_normalization, replay_buffer_stats_dict, task_list, tasks, train_tasks, test_tasks)
