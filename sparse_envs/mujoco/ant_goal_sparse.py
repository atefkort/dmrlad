import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import colorsys



class AntGoalSparseEnv(RandomEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.meta_mode = 'train'
        self.termination_possible = kwargs.get('termination_possible', True)
        self.steps = 0
        self.goal = {'goal': np.array([0.0, 0.0]), 'angle': 0.0, 'radius': 0.0}
        self.task_max_radius = kwargs.get('task_max_radius', 1.0)

        RandomEnv.__init__(self, kwargs.get('log_scale_limit', 0), 'ant.xml', 5, hfield_mode=kwargs.get('hfield_mode', 'gentle'), rand_params=[])
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, action):
        try:
            self.do_simulation(action, self.frame_skip)
        except:
            raise RuntimeError("Simulation error, common error is action = nan")
        xposafter = np.array(self.get_body_com("torso"))
        ob = self._get_obs()

        # Tuned like in PEARL
        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal['goal']))
        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        dist = np.linalg.norm(xposafter[:2] - self.goal['goal']) + 4.0
        sparse_goal_reward = goal_reward
        if dist > 0.8:
            sparse_goal_reward = -np.sum(np.abs(self.goal['goal'])) + 4.0
        sparse_reward = sparse_goal_reward - ctrl_cost - contact_cost + survive_reward
        reward = sparse_reward
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0  # original gym values: 0.2 and 1.0
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(
            reward_goal=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            xposafter=xposafter,
            done=done,
            true_task=dict(base_task=0, specification=self.goal['goal'])
        )

    # from pearl rlkit
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        # for time variant
        self.steps = 0

        # reset velocity to task starting velocity
        self.goal = self._task
        self.recolor()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20
        
    def change_goal(self):
        if self.meta_mode == 'train':
            self.goal = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self.goal = np.random.choice(self.test_tasks)

        self.recolor()
        self.steps = 0

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        hue = self.goal['angle'] / (2 * np.pi) # maps color in hsv color space
        saturation = self.goal['radius'] / (self.task_max_radius)
        rgb_value_tuple = colorsys.hsv_to_rgb(hue,saturation,1)
        geom_rgba[1:, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba
        
    def sample_tasks(self, num_tasks):
        #np.random.seed(1337)
        a = np.random.random(num_tasks) * 2 * np.pi
        r = self.task_max_radius * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

        #plt.figure()
        #plt.scatter(goals[:,0], goals[:,1])
        #plt.show()
        tasks = [{'goal': goal, 'angle': angle, 'radius': rv} for goal, angle, rv in zip(goals, a, r)]
        return tasks

    def set_meta_mode(self, mode):
        self.meta_mode = mode

    def reward(self, info, goal):
        reward_ctrl, reward_contact = info["reward_ctrl"], info["reward_contact"]
        reward_survive, xposafter = info["reward_survive"], info["xposafter"]
        done = info["done"]
        dist = np.linalg.norm(xposafter[:2] - goal)
        goal_reward = -np.sum(np.abs(xposafter[:2] - goal)) + 4.0
        sparse_goal_reward = goal_reward
        if dist > 0.8:
            sparse_goal_reward = -np.sum(np.abs(goal)) + 4.0
        sparse_reward = sparse_goal_reward + reward_ctrl + reward_contact + reward_survive

        return sparse_reward, done
    
    def get_train_goals(self):
        return [task["goal"] for task in self.train_tasks]


if __name__ == "__main__":

    env = HalfCheetahChangingVelEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(2000):
            env.render()
            env.step(env.action_space.sample())  # take a random action
