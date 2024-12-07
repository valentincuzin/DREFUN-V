from copy import deepcopy
from DREFUN import DREFUN
from RLalgo import PolitiqueDirectSearch
import gymnasium as gym
from utils import eval_politique, plot_sumrwdperepi

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy_raw = PolitiqueDirectSearch(4, 2)
    drefun = DREFUN(policy_raw, env)

    reward_func = drefun.generate_reward_function(
        task_description="""Balance a pole on a cart, 
        Num Observation Min Max
        0 Cart Position -4.8 4.8
        1 Cart Velocity -Inf Inf
        2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
        3 Pole Angular Velocity -Inf Inf
        Since the goal is to keep the pole upright for as long as possible, by default, a reward of +1 is given for every step taken, including the termination step. The default reward threshold is 500 for v1
        """,
    )

    print(reward_func)

    policy_test = deepcopy(policy_raw)
    raw_res, _, succrate_res = policy_raw.train(env)
    res, _, succrate = policy_test.train(env, reward_func)

    # plot_sumrwdperepi(raw_res)
    # plot_sumrwdperepi(res)

    print("Success rate without reward function:", succrate_res)
    print("Success rate with reward function:", succrate)
