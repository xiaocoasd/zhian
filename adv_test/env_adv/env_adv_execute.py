from abc import ABC
from tianshou.data import Batch, to_numpy

# 执行模块：
# --在该模块中做改变属性和奖励结构异化的执行
# ----需要更改环境属性，即需要相关的属性
# --只做一回合的执行
# ----只需要一小部分。
# 使用第二种


class EnvAdvExe(ABC):

    def __init__(
        self,
        policy,
        is_rdm,
        env,
        data,
    ) -> None:
        super().__init__()

        self.policy = policy
        self.is_rdm = is_rdm
        self.env = env
        self.data = data

    def run(self):
        done = False
        frames = []
        rew_t = 0.0

        while True:
            last_state = self.data.policy.pop("hidden_state", None)

            frames.append(self.env.render())
            if self.is_rdm:
                rdm_act = [self.env_adv.action_space[0].sample()]
                rdm_act_sample = self.policy.map_action_inverse(rdm_act)
                self.data.update(act=rdm_act_sample)
            else:
                result = self.policy(self.data, last_state)
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", Batch())
                if state is not None:
                    policy.hidden_state = state

                act = to_numpy(result.act)
                self.data.update(policy=result.get("policy", Batch()), act=act)

            obs_next, rew, terminated, truncated, info = self.env.step(
                action=self.data.act
                # is_change_reward=False,
            )

            done = terminated | truncated
            # 更新数据
            self.data.update(
                obs=obs_next,
                rew=rew,
                done=done,
                info=info,
            )

            rew_t += rew

            if any(done):
                break

        #     print(rew_t)
        # print("end rew:{}".format(rew_t))
        flat_frames = [sublist[0] for sublist in frames]
        return flat_frames, rew_t
