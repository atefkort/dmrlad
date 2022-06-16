from sparse_envs.mujoco.hopper_sparse_velocity import HopperSparseVelocityEnv
from . import register_env


@register_env('hopper-sparse-vel')
class HopperSparseVelWrappedEnv(HopperSparseVelocityEnv):
    def __init__(self, *args, **kwargs):
        HopperSparseVelocityEnv.__init__(self, *args, **kwargs)
