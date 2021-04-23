import pytest
from ray.rllib import RolloutWorker
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.experiment.data import DataModuleSpec, TrajectoryData, TransitionData
from lqsvg.experiment.worker import make_worker
from lqsvg.testing.fixture import standard_fixture

n_state = standard_fixture((2, 3), "NState")
n_ctrl = standard_fixture((2, 3), "NCtrl")
horizon = standard_fixture((1, 7, 10), "Horizon")


@pytest.fixture
def worker(n_state: int, n_ctrl: int, horizon: int) -> RolloutWorker:
    with nt.suppress_named_tensor_warning():
        worker = make_worker(
            env_config=dict(
                n_state=n_state, n_ctrl=n_ctrl, horizon=horizon, num_envs=2
            ),
            log_level="WARNING",
        )
    return worker


def test_trajectory_data(
    worker: RolloutWorker, n_state: int, n_ctrl: int, horizon: int
):
    data_spec = DataModuleSpec(total_trajs=100)
    datamodule = TrajectoryData(worker, data_spec)
    datamodule.build_dataset(prog=False)
    datamodule.setup()

    for loader in (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ):
        batch_size = min(datamodule.spec.batch_size, len(loader.dataset))
        batch = next(iter(loader))
        assert len(batch) == 3
        obs, act, new_obs = batch

        def tensor_info(tensor: Tensor, dim: int, batch_size: int = batch_size) -> str:
            return f"{tensor.shape}, B: {batch_size}, H: {horizon}, dim: {dim}"

        assert obs.shape == (batch_size, horizon, n_state + 1), tensor_info(
            obs, n_state
        )
        assert act.shape == (batch_size, horizon, n_ctrl), tensor_info(act, n_ctrl)
        assert new_obs.shape == (batch_size, horizon, n_state + 1), tensor_info(
            new_obs, n_state
        )


def test_transition_data(worker: RolloutWorker, n_state: int, n_ctrl: int):
    data_spec = DataModuleSpec(total_trajs=100)
    datamodule = TransitionData(worker, data_spec)
    datamodule.build_dataset(prog=False)
    datamodule.setup()

    for loader in (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ):
        batch_size = min(datamodule.spec.batch_size, len(loader.dataset))
        batch = next(iter(loader))
        assert len(batch) == 3
        obs, act, new_obs = batch

        def tensor_info(tensor: Tensor, dim: int, batch_size: int = batch_size) -> str:
            return f"{tensor.shape}, B: {batch_size}, dim: {dim}"

        assert obs.shape == (batch_size, n_state + 1), tensor_info(obs, n_state)
        assert act.shape == (batch_size, n_ctrl), tensor_info(act, n_ctrl)
        assert new_obs.shape == (batch_size, n_state + 1), tensor_info(new_obs, n_state)
