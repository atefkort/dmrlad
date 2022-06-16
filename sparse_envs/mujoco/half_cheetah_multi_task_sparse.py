import colorsys
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from meta_rand_envs.base import NonStationaryMetaEnv


class HalfCheetahSparseMultiTaskEnv(NonStationaryMetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.task_variants = kwargs.get('task_variants', ['forward velocity', 'backward velocity', 'front goal', 'back goal'])
        self.termination_possible = kwargs.get('termination_possible', False)
        self.current_task = None
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        self.active_task = {'base_task': 1, 'specification': 1, 'color': np.array([0,1,0])}
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        # should actually go into NonStationaryGoalVelocityEnv, breaks abstraction
        self._init_geom_rgba = self.model.geom_rgba.copy()

        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks
        self.reset_task(0)

    def step(self, action):
        self.check_env_change()

        xposbefore = self.sim.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos.copy()
        ob = self._get_obs()

        if self.active_task['base_task'] in [0, 1]: #velocity tasks
            forward_vel = (xposafter[0] - xposbefore[0]) / self.dt
            reward_run = -1.0 * abs(forward_vel - self.active_task['specification'])
            sparse_reward = self.sparsify_rewards(reward_run, goal_radius=0.1)
        elif self.active_task['base_task'] in [2, 3]: # goal tasks
            reward_run = -1.0 * abs(xposafter[0] - self.active_task['specification']) 
            sparse_reward = self.sparsify_rewards(reward_run, goal_radius=0.5)         
        else:
            raise RuntimeError("bask task not recognized")
        
        reward_ctrl = -0.5 * np.sum(np.square(action))
        reward = reward_ctrl + sparse_reward  
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run,
                                      reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=self.active_task['base_task'],
                                                     specification=self.active_task['specification']),
                                      velocity=(xposafter[0] - xposbefore[0]) / self.dt,
                                      position=xposafter[0])

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
        self.active_task = self.task
        self.reset_change_points()
        self.recolor()
        self.steps = 0
        self.reset()

    def sample_tasks(self, num_tasks):
        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = int(num_tasks / num_base_tasks)
        num_tasks_per_subtask_half = int(num_tasks_per_subtask / 2)
        np.random.seed(1337)

        tasks = []
        #velocity tasks
        if 'forward velocity' in self.task_variants:
            velocities = np.linspace(0.0, 3.0, num_tasks_per_subtask)
            tasks_velocity = [{'base_task': 0, 'specification': velocity, 'color': np.array([1,0,0])} for velocity in velocities]
            tasks += (tasks_velocity)

        if 'bacward velocity' in self.task_variants:
            velocities = np.linspace(-3.0, 0.0, num_tasks_per_subtask)
            tasks_velocity = [{'base_task': 1, 'specification': velocity, 'color': np.array([0,1,0])} for velocity in velocities]
            tasks += (tasks_velocity)

        # goal tasks
        if 'front goal' in self.task_variants:
            goals = np.random.uniform(0, 5, size=(num_tasks_per_subtask))
            tasks_goal = [{'base_task': 2, 'specification': goal, 'color': np.array([0,0,1])} for goal in goals]
            tasks += (tasks_goal)

        if 'back goal' in self.task_variants:
            goals = np.random.uniform(-5, 0, size=(num_tasks_per_subtask))
            tasks_goal = [{'base_task': 3, 'specification': goal, 'color': np.array([0.5,0.5,0])} for goal in goals]
            tasks += (tasks_goal)

        return tasks

    def change_active_task(self, step=100, dir=1):
        if self.meta_mode == 'train':
            self.active_task = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self.active_task = np.random.choice(self.test_tasks)
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value = self.active_task['color']
        geom_rgba[1:, :3] = np.asarray(rgb_value)
        self.model.geom_rgba[:] = geom_rgba

    def get_train_goals(self):
        return [[task["base_task"], task["specification"]] for task in self.train_tasks]

    def sparsify_rewards(self, r, goal_radius = 0.1):
        ''' zero out rewards when outside the goal radius '''
        
        if r < - goal_radius:
            r = -4
        r = r + 2
        return r

    def reward(self, info, goal):
        reward_ctrl, vel, pos = info["reward_ctrl"], info["velocity"], info["position"]
        if goal[0] in [0,1]:
            reward = -1.0 * abs(vel - goal[1])
        else:
            reward = -1.0 * abs(pos - goal[1])      
        sparse_reward = self.sparsify_rewards(reward)
        return sparse_reward + reward_ctrl, False