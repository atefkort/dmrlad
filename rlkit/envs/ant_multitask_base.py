import numpy as np

from rlkit.envs.ant import AntEnv

# from gym.envs.mujoco.ant import AntEnv


class MultitaskAntEnv(AntEnv):
    def __init__(self, **kwargs):
        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks
        self._goal = self.tasks[0]["goal"]
        AntEnv.__init__(self)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task[
            "goal"
        ]  # assume parameterization of task by single vector
        self.reset()

    def get_train_goals(self):
        return [task["goal"] for task in self.train_tasks]
