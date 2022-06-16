from sparse_envs.mujoco.half_cheetah_multi_task_sparse import HalfCheetahSparseMultiTaskEnv
from . import register_env


@register_env('cheetah-sparse-multi-task')
class HalfCheetahMultiTaskSparseWrappedEnv(HalfCheetahSparseMultiTaskEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahSparseMultiTaskEnv.__init__(self, *args, **kwargs)
