import colorsys
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from meta_rand_envs.base import NonStationaryGoalVelocityEnv

class HalfCheetahSparseGoalEnv(NonStationaryGoalVelocityEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.termination_possible = kwargs.get('termination_possible', False)
        NonStationaryGoalVelocityEnv.__init__(self, *args, **kwargs)
        
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xpos = self.sim.data.qpos[0]
        ob = self._get_obs()
        dist = -1.0 * abs(xpos - self.active_task)
        sparse_reward = self.sparsify_rewards(dist)
        reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        sparse_reward = sparse_reward + reward_ctrl
        reward = sparse_reward
        
        success = float(abs(dist) <= 0.05)
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(success = success, reward_run=dist, reward_ctrl=reward_ctrl,sparse_reward=sparse_reward,
                                      true_task=dict(base_task=0, specification=self.active_task), xpos=xpos)

    # from pearl
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task['goal']
        self.reset_change_points()
        self.recolor()
        self.steps = 0
        self.reset()

    def reward(self, info, goal):
        reward_ctrl, forward_vel = info["reward_ctrl"], info["xpos"]
        forward_reward = -1.0 * abs(forward_vel - goal)
        sparse_reward = self.sparsify_rewards(forward_reward)
        return sparse_reward + reward_ctrl, False
    
    def get_train_goals(self):
        return [task["goal"] for task in self.train_tasks]

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        goal_radius = 0.5
        if r < - goal_radius:
            r = -4
        r = r + 2
        return r

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        goals = np.random.uniform(1, 6, size=(num_tasks,))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_meta_mode(self, mode):
        self.meta_mode = mode