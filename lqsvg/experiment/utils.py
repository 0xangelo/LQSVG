"""Utilities for RLlib sample batches, warnings, and linear feedback policies."""
import datetime
import functools
import itertools
import operator
import os
from collections import defaultdict
from numbers import Number
from typing import Callable, Iterable, List, Optional

import pandas as pd
import wandb
from ray.tune.utils import flatten_dict

from lqsvg.types import Directory


def directories_with_target_files(
    directories: List[str], istarget: Callable[[str], bool]
) -> List[Directory]:
    """Return directories that havy any file of interest."""
    # (path, subpath, files) for all dirs
    all_paths = itertools.chain(*map(os.walk, directories))
    # convert to directory type
    all_dirs = itertools.starmap(Directory, all_paths)
    # entries that have >0 files (may be unnecessary)
    nonempty = filter(operator.attrgetter("files"), all_dirs)
    # entries that have target files
    with_target_files = (x for x in nonempty if any(map(istarget, x.files)))
    return list(with_target_files)


def experiment_directories(rootdir: str) -> List[Directory]:
    """Return experiment directories."""
    return directories_with_target_files(
        [rootdir], lambda f: f.startswith("progress") and f.endswith(".csv")
    )


def crashed_experiments(rootdir: str) -> List[Directory]:
    """Return experiment directories that have crash logs."""
    exp_dirs = experiment_directories(rootdir)
    return [
        d
        for d in exp_dirs
        if any(map(functools.partial(operator.eq, "error.txt"), d.files))
    ]


def filtered_wandb_runs(path: str, filters: dict, state: str = "finished"):
    """Returns wandb run objects matching the given filters and state."""
    api = wandb.Api()
    return (run for run in api.runs(path, filters=filters) if run.state == state)


def wandb_runs_dataframe(
    path: str,
    configs: dict,
    tags: Iterable[str] = (),
    filters: Optional[dict] = None,
    state: str = "finished",
) -> pd.DataFrame:
    """Retrieve data from experiments with given tags as a dataframe."""
    filters = defaultdict(list, filters or {})
    filters["$and"] += [{"tags": tag} for tag in tags]
    filters["$and"] += [{"config." + k: v for k, v in configs.items()}]

    def col_value(val):
        if isinstance(val, (str, Number)):
            return val
        return str(val)

    dfs = []
    for run in filtered_wandb_runs(path, filters, state):
        dataframe = run.history()

        # Merge run history and configurations
        for key, val in flatten_dict(run.config).items():
            dataframe["config/" + key] = col_value(val)

        # Merge run history and summary statistics
        for key, val in run.summary.items():
            if key.startswith("_"):
                continue
            dataframe["summary/" + key] = val

        dfs += [dataframe]

    return pd.concat(dfs, ignore_index=True)


def calver() -> str:
    """Return a standardized version number using CalVer."""
    today = datetime.date.today()
    return f"{today.month}.{today.day}.0"
