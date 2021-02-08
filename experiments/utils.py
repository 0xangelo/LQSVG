"""Utilities for handling sample batches in RLlib."""
import numpy as np
from ray.rllib import SampleBatch


def group_batch_episodes(samples: SampleBatch) -> SampleBatch:
    """Return the sample batch with rows grouped by episode id.

    Moreover, rows are sorted by timestep.

    Warning:
        Modifies the sample batch in-place
    """
    # Assume "t" is the timestep key in the sample batch
    sort_ts_idx = np.argsort(
        samples["t"]
    )  # line too loooooooooooooooooooooooooooooooooooooooooooooooong
    for key, val in samples.items():
        samples[key] = val[sort_ts_idx]

    # Stable sort is important so that we don't alter the order
    # of timesteps
    sort_eps_idx = np.argsort(samples[SampleBatch.EPS_ID], kind="stable")
    for key, val in samples.items():
        samples[key] = val[sort_eps_idx]

    return samples


def num_complete_episodes(samples: SampleBatch) -> int:
    """Return the number of complete episodes in a SampleBatch."""
    num_eps = len(np.unique(samples[SampleBatch.EPS_ID]))
    num_dones = np.sum(samples[SampleBatch.DONES])
    assert num_dones <= num_eps, (num_dones, num_eps)
    return num_dones
