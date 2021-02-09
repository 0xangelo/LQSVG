# pylint:disable=missing-docstring,invalid-name
from ray.rllib import RolloutWorker

from lqsvg.envs.lqr.gym import RandomVectorLQG
from lqsvg.policy.time_varying_linear import LQGPolicy

# from lqsvg.policy import RandomPolicy


def make_worker(num_envs: int = 1) -> RolloutWorker:
    # pylint:disable=import-outside-toplevel
    import lqsvg
    from raylab.envs import get_env_creator

    lqsvg.register_all()

    # Create and initialize
    n_state, n_ctrl, horizon = 2, 2, 100
    seed = 42
    worker = RolloutWorker(
        env_creator=get_env_creator("RandomVectorLQG"),
        env_config={
            "n_state": n_state,
            "n_ctrl": n_ctrl,
            "horizon": horizon,
            "gen_seed": seed,
            "num_envs": num_envs,
        },
        num_envs=num_envs,
        policy_spec=LQGPolicy,
        # policy_spec=RandomPolicy,
        policy_config={},
        rollout_fragment_length=horizon,
        batch_mode="truncate_episodes",
        _use_trajectory_view_api=False,
    )
    assert isinstance(worker.env, RandomVectorLQG)
    assert worker.async_env.num_envs == num_envs
    worker.foreach_trainable_policy(lambda p, _: p.initialize_from_lqg(worker.env))
    return worker


def test_worker():
    # pylint:disable=import-outside-toplevel
    import numpy as np

    worker = make_worker(4)
    samples = worker.sample()
    print("Count:", samples.count)
    idxs = np.random.permutation(samples.count)[:10]
    for k in samples.keys():
        print("Key:", k)
        print(samples[k][idxs])
        print(samples[k][idxs].dtype)


if __name__ == "__main__":
    test_worker()
