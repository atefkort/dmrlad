from sparse_envs.mujoco.walker2d_sparse_velocity import Walker2DSparseVelocityEnv
from . import register_env


@register_env('walker2d-sparse-vel')
class Walker2DSparseVelWrappedEnv(Walker2DSparseVelocityEnv):
    def __init__(self, *args, **kwargs):
        Walker2DSparseVelocityEnv.__init__(self, *args, **kwargs)
