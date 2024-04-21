import sys
from adv_test.env_adv.env_adv_map.env_adv_CartPole import EnvCartPoleAdv


def env_adv_make(env_name, render_mode, is_change_reward, env=None):
    if env is None:
        if env_name == "CartPole-v1":
            return EnvCartPoleAdv(
                is_change_reward=is_change_reward,
                render_mode=render_mode,
            )
        elif env_name == "Pendulum-v1":
            print("未实现")
            sys.exit()
        else:
            print("未实现")
            sys.exit()
    else:
        print("目前未支持")
        sys.exit()
