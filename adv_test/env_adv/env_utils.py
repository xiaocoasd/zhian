import sys
from adv_test.env_adv.env_adv_map.env_adv_CartPole import EnvCartPoleAdv

from adv_test.env_adv.env_adv_map.env_adv_Highway import EnvHighwayAdv

from adv_test.env_adv.env_adv_map.env_adv_Pendulum import EnvPendulumAdv


def env_adv_make(env_name, render_mode, is_change_reward, args, env=None):
    if env is None:
        if env_name == "CartPole-v1":
            return EnvCartPoleAdv(
                is_change_reward=is_change_reward,
                render_mode=render_mode,
                args=args,
            )
        elif env_name == "Pendulum-v1":
            return EnvPendulumAdv(
                is_change_reward=is_change_reward,
                render_mode=render_mode,
                args=args,
            )
            sys.exit()
        elif env_name == "highway-v0":

            env_t = EnvHighwayAdv(
                is_change_reward=is_change_reward,
                render_mode=render_mode,
                args=args,
            )
            env_t.unwrapped.configure({"offscreen_rendering": True})

            return env_t
    else:
        print("目前未支持")
        sys.exit()
