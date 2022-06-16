from sparse_envs.mujoco.half_cheetah_goal_sparse import HalfCheetahSparseGoalEnv
from . import register_env


@register_env('cheetah-goal-sparse')
class HalfCheetahSparseGoalWrappedEnv(HalfCheetahSparseGoalEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahSparseGoalEnv.__init__(self, *args, **kwargs)
