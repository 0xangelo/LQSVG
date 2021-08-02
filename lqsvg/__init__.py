"""Research on Stochastic Value Gradient methods in LQGs."""
__author__ = """Ângelo Gregório Lovatto"""
__email__ = "angelolovatto@gmail.com"
__version__ = "0.1.0"


def register_all_environments():
    """Register all custom environments in Tune."""
    from ray.tune import register_env

    from lqsvg.envs.registry import ENVS

    for name, creator in ENVS.items():
        register_env(name, creator)
