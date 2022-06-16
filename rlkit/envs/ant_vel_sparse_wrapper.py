from sparse_envs.mujoco.ant_velocity_sparse import AntSparseVelocityEnv
from . import register_env


@register_env('ant-vel-sparse')
class AntSparseVelWrappedEnv(AntSparseVelocityEnv):
    def __init__(self, *args, **kwargs):
        AntSparseVelocityEnv.__init__(self, *args, **kwargs)
