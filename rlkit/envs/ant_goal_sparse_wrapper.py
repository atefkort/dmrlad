import numpy as np
from sparse_envs.mujoco.ant_goal_sparse import AntGoalSparseEnv

from . import register_env


@register_env('ant-goal-sparse')
class AntGoalSparseWrappedEnv(AntGoalSparseEnv):
    def __init__(self, *args, **kwargs):
        super(AntGoalSparseWrappedEnv, self).__init__(*args, **kwargs)
        self.tasks = self.sample_tasks(kwargs['n_train_tasks']+kwargs['n_eval_tasks'])
        self.train_tasks = self.tasks[:kwargs['n_train_tasks']]
        self.test_tasks = self.tasks[kwargs['n_train_tasks']:]
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.goal = self._task
        self.recolor()
        self.reset()
