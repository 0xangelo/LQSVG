# pylint:disable=missing-docstring,invalid-name
from ray.rllib import RolloutWorker

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.gym import RandomVectorLQG
from lqsvg.policy.time_varying_linear import LQGPolicy


def make_worker(env_config: dict) -> RolloutWorker:
    # pylint:disable=import-outside-toplevel
    import lqsvg
    from raylab.envs import get_env_creator

    lqsvg.register_all()

    # Create and initialize
    num_envs = env_config["num_envs"]
    worker = RolloutWorker(
        env_creator=get_env_creator("RandomVectorLQG"),
        env_config=env_config,
        num_envs=num_envs,
        policy_spec=LQGPolicy,
        policy_config={"framework": "torch"},
        rollout_fragment_length=env_config["horizon"],
        batch_mode="truncate_episodes",
        _use_trajectory_view_api=False,
    )
    assert isinstance(worker.env, RandomVectorLQG)
    assert worker.async_env.num_envs == num_envs
    worker.foreach_trainable_policy(lambda p, _: p.initialize_from_lqg(worker.env))
    return worker


@nt.suppress_named_tensor_warning()
def test_worker():
    # pylint:disable=import-outside-toplevel
    import numpy as np

    worker = make_worker(
        dict(n_state=2, n_ctrl=2, horizon=100, num_envs=4, gen_seed=42)
    )
    samples = worker.sample()
    print("Count:", samples.count)
    idxs = np.random.permutation(samples.count)[:10]
    for k in samples.keys():
        print("Key:", k)
        print(samples[k][idxs])
        print(samples[k][idxs].dtype)


if __name__ == "__main__":
    test_worker()
