"""A loader for the dataset of spreading potentials."""

import pathlib
import re

import pandas as pd


def get_simulation_params(file_name: pathlib.Path) -> tuple[str, str, str]:
    pattern = r"proto-(OR|AND)--p-([\d.]+)--net-(.*)\.csv"
    return re.match(pattern, file_name.name).groups()


def read_csv(file_path: pathlib.Path) -> pd.DataFrame:
    proto, p, net = get_simulation_params(file_path)
    file_df = pd.read_csv(file_path)
    file_df["network"] = net
    file_df["protocol"] = proto
    file_df["p"] = p
    return file_df
