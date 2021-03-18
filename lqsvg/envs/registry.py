"""Collects and registers custom Gym environments for Ray Tune."""
import gym
from ray.rllib import VectorEnv

ENVS = {}


def _random_lqg_maker(config: dict) -> gym.Env:
    from lqsvg.envs.lqr.gym import LQGGenerator, RandomLQGEnv

    spec = LQGGenerator(**config)
    return RandomLQGEnv(spec)


def _random_vector_lqg_maker(config: dict) -> VectorEnv:
    from lqsvg.envs.lqr.gym import LQGGenerator, RandomVectorLQG

    spec = LQGGenerator(**config)
    return RandomVectorLQG(spec)


ENVS.update(
    {
        "RandomLQG": _random_lqg_maker,
        "RandomVectorLQG": _random_vector_lqg_maker,
    }
)
