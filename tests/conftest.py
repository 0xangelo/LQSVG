import pytest

from .mock_env import MockEnv
from .mock_env import MockReward
from .mock_env import MockTermination


# ==============================================================================
# Test setup from:
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
def pytest_addoption(parser):
    parser.addoption(
        "--skipslow", action="store_true", default=False, help="skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipslow"):
        # --skipslow given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="--skipslow option passed")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
# ==============================================================================


@pytest.fixture(autouse=True, scope="session")
def init_ray():
    import ray

    ray.init()
    yield
    ray.shutdown()


@pytest.fixture(autouse=True, scope="session")
def disable_gym_logger_warnings():
    import logging
    import gym

    gym.logger.set_level(logging.ERROR)


@pytest.fixture(autouse=True, scope="session")
def register_envs():
    import lqsvg
    import raylab
    import raylab.envs as envs

    lqsvg.register_all_environments()
    raylab.register_all_environments()
    # pylint:disable=unnecessary-lambda
    envs.register_env("MockEnv", lambda c: MockEnv(c))
    envs.register_reward_fn("MockEnv")(MockReward)
    envs.register_termination_fn("MockEnv")(MockTermination)
