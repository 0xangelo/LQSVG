# pylint:disable=all
import numpy as np
from ray.rllib import SampleBatch
from tqdm.auto import tqdm


def collect_with_progress(worker, total_trajs, prog: bool = True) -> SampleBatch:
    with tqdm(
        total=total_trajs, desc="Collecting", unit="traj", disable=not prog
    ) as pbar:
        sample_batch: SampleBatch = worker.sample()
        eps = num_complete_episodes(sample_batch)
        while eps < total_trajs:
            old_eps = eps
            sample_batch = sample_batch.concat(worker.sample())
            eps = num_complete_episodes(sample_batch)
            pbar.update(eps - old_eps)

    return sample_batch


def num_complete_episodes(samples: SampleBatch) -> int:
    """Return the number of complete episodes in a SampleBatch."""
    num_eps = len(np.unique(samples[SampleBatch.EPS_ID]))
    num_dones = np.sum(samples[SampleBatch.DONES]).item()
    assert (
        num_dones <= num_eps
    ), f"More done flags than episodes: dones={num_dones}, episodes={num_eps}"
    return num_dones
