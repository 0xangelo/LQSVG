# pylint:disable=missing-docstring
from torch import Tensor
from torch.utils.data import Dataset

from experiments.ch5.model import Batch
from lqsvg.torch import named as nt


class TrajectorySegmentDataset(Dataset):
    # noinspection PyArgumentList
    def __init__(self, obs: Tensor, act: Tensor, new_obs: Tensor, segment_len: int):
        # Pytorch Lightning deepcopies the dataloader when using overfit_batches=True
        # Deepcopying is incompatible with named tensors for some reason
        self.tensors = nt.unnamed(
            *(x.align_to("B", "H", ...) for x in (obs, act, new_obs))
        )
        self.segment_len = segment_len
        horizon: int = obs.size("H")
        trajs: int = obs.size("B")
        self.segs_per_traj = horizon - segment_len + 1
        self._len = trajs * self.segs_per_traj

    def __getitem__(self, index) -> Batch:
        traj_idx = index // self.segs_per_traj
        timestep_start = index % self.segs_per_traj
        # noinspection PyTypeChecker
        return tuple(
            t[traj_idx, timestep_start : timestep_start + self.segment_len]
            for t in self.tensors
        )

    def __len__(self) -> int:
        return self._len
