from sparse_envs.mujoco.half_cheetah_velocity_sparse import HalfCheetahSparseVelocityEnv
from . import register_env


@register_env('cheetah-vel-sparse')
class HalfCheetahNonStationaryVelWrappedEnv(HalfCheetahSparseVelocityEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahSparseVelocityEnv.__init__(self, *args, **kwargs)
