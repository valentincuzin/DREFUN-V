from log.log_config import init_logger
from logging import getLogger
from DREFUN import DREFUN
from RLalgo import PolitiqueDirectSearch
import gymnasium as gym

if __name__ == "__main__":  # TODO parametrer le mode debug
    init_logger()
    logger = getLogger("DREFUN")
    env = gym.make("CartPole-v1")
    learning_method = PolitiqueDirectSearch(env)
    drefun = DREFUN(learning_method, env)

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

    drefun.evaluate_policy()
