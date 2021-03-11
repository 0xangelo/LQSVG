# pylint:disable=all
from ray.rllib import SampleBatch
from tqdm.auto import tqdm

from utils import num_complete_episodes  # pylint:disable=wrong-import-order


def collect_with_progress(worker, total_trajs) -> SampleBatch:
    with tqdm(total=total_trajs, desc="Collecting", unit="traj") as pbar:
        sample_batch: SampleBatch = worker.sample()
        eps = num_complete_episodes(sample_batch)
        while eps < total_trajs:
            old_eps = eps
            sample_batch = sample_batch.concat(worker.sample())
            eps = num_complete_episodes(sample_batch)
            pbar.update(eps - old_eps)

    return sample_batch
