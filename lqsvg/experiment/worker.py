"""Utilities for standard rollout worker creation for LQG environments."""
# pylint:disable=invalid-name
import numpy as np
from ray.rllib import RolloutWorker
from ray.rllib.env import EnvContext
from ray.rllib.utils.typing import EnvType
from raylab.envs import get_env_creator

import lqsvg
import lqsvg.torch.named as nt
from lqsvg.envs.lqr.gym import RandomVectorLQG
from lqsvg.policy.time_varying_linear import LQGPolicy


def make_worker(*, env_config: dict, **kwargs) -> RolloutWorker:
    """Build rollout worker for a linear feedback policy on LQG.

    This function calls `initialize_from_lqg` on the created policy with the
    created LQG instance as an argument.
    The worker will sample complete trajectories on each call to .sample().

    Args:
        env_config: the configuration for the LQG environment

    Returns:
        A RolloutWorker instance
    """
    _validate_env = kwargs.pop("validate_env", None)

    def validate_env(env: EnvType, env_context: EnvContext):
        assert isinstance(env, RandomVectorLQG)
        assert env.num_envs == env_context["num_envs"]

        if _validate_env:
            _validate_env(env, env_context)

    policy_config = kwargs.pop("policy_config", {})
    policy_config["framework"] = "torch"

    # Create and initialize
    worker = RolloutWorker(
        env_creator=get_env_creator("RandomVectorLQG"),
        validate_env=validate_env,
        env_config=env_config,
        num_envs=env_config["num_envs"],
        policy_spec=LQGPolicy,
        policy_config=policy_config,
        rollout_fragment_length=env_config["horizon"],
        batch_mode="truncate_episodes",
        _use_trajectory_view_api=False,
        **kwargs
    )
    worker.foreach_trainable_policy(lambda p, _: p.setup(worker.env))
    return worker


@nt.suppress_named_tensor_warning()
def test_worker():
    # pylint:disable=missing-function-docstring
    lqsvg.register_all()
    worker = make_worker(
        env_config=dict(n_state=2, n_ctrl=2, horizon=100, num_envs=4, seed=42)
    )
    samples = worker.sample()
    print("Count:", samples.count)
    idxs = np.random.permutation(samples.count)[:10]
    for k in samples.keys():
        print("Key:", k)
        print(samples[k][idxs])
        print(samples[k][idxs].dtype)
        print()


if __name__ == "__main__":
    test_worker()
