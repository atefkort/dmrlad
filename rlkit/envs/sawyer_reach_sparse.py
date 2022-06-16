from sparse_envs.mujoco.sawyer_reach import SawyerReachEnv
from . import register_env


@register_env('sawyer-reach-sparse')
class SawyerReachEnvWrappedEnv(SawyerReachEnv):
    def __init__(self, *args, **kwargs):
        SawyerReachEnv.__init__(self, *args, **kwargs)
