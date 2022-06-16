from sparse_envs.mujoco.sawyer_push import SawyerPushEnv
from . import register_env


@register_env('sawyer-push-sparse')
class SawyerPushEnvWrappedEnv(SawyerPushEnv):
    def __init__(self, *args, **kwargs):
        SawyerPushEnv.__init__(self, *args, **kwargs)
