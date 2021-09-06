"""Utilities for RLlib sample batches, warnings, and linear feedback policies."""
import datetime
import functools
import itertools
import operator
import os
from typing import Callable, Iterable, List

import pandas as pd
import wandb

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


def tagged_experiments_dataframe(tags: Iterable[str]) -> pd.DataFrame:
    """Retrieve data from experiments with given tags as a dataframe."""
    api = wandb.Api()
    runs = api.runs(
        "angelovtt/LQG-SVG", filters={"$and": [{"tags": tag} for tag in tags]}
    )
    dfs = (run.history() for run in runs)
    return pd.concat(dfs, ignore_index=True)


def calver() -> str:
    """Return a standardized version number using CalVer."""
    today = datetime.date.today()
    return f"{today.month}.{today.day}.0"
