"""A loader for the dataset of spreading potentials."""

import re
from pathlib import Path

import pandas as pd
from nsl_data_utils.loaders.constants import (
    ACTOR,
    EXPOSED,
    MLN_SP_DATA_PATH,
    NOT_EXPOSED,
    P,
    PEAK_INFECTED,
    PEAK_ITERATION,
    SIMULATION_LENGTH,
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
    ARTIFICIAL_SMALL,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    ARXIV_NETSCIENCE_COAUTHORSHIP_MATH,
    AUCS,
    CANNES,
    CKM_PHYSICIANS,
    EU_TRANSPORTATION,
    FMRI74,
    L2_COURSE,
    LAZEGA,
    TIMIK1Q2009,
    TOY_NETWORK,
)


def get_simulation_params(file_name: Path) -> tuple[str, str, str]:
    pattern = r"proto-(OR|AND)--p-([\d.]+)--net-(.*)\.csv"
    return re.match(pattern, file_name.name).groups()


def read_csv(file_path: Path) -> pd.DataFrame:
    proto, p, net = get_simulation_params(file_path)
    file_df = pd.read_csv(file_path)
    file_df["network"] = net
    file_df["protocol"] = proto
    file_df["p"] = p
    return file_df


def load_sp(csv_paths: list[str]) -> pd.DataFrame:
    """Read spreading potentials stored in files indicated by given regex."""
    raw_csvs = []
    for _, file_path in enumerate(csv_paths):
        raw_csvs.append(read_csv(file_path))
    sp = pd.concat(raw_csvs, axis=0, ignore_index=True)
    assert len(sp["network"].unique()) == 1
    sp[[SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION, P]] = sp[
        [SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION, P]
    ].apply(pd.to_numeric)
    sp[ACTOR] = sp[ACTOR].astype(str)
    return sp


def _get_csv_paths(csv_regex: str) -> list[Path]:
    return list(Path(MLN_SP_DATA_PATH).glob(csv_regex))


def _sort_csv_paths(csv_regex: str) -> dict[str, list[Path]]:
    """Reads all CSVs from a directory and sort it by th network's name."""
    all_paths = _get_csv_paths(csv_regex=csv_regex)
    sorted_paths = {}
    for csv_path in all_paths:
        net_name = csv_path.stem.split("-")[-1]
        if not sorted_paths.get(net_name):
            sorted_paths[net_name] = [csv_path]
        else:
            sorted_paths[net_name].append(csv_path)
    return sorted_paths


def _sp_not_implemented():
    raise NotImplementedError(f"Spreading potentials have been not prepared yet!")


def load_sp_paths(net_type: str, net_name: str) -> list[Path]:
    """Load spreading potentials dataset for given network."""
    if net_type in {ARTIFICIAL_ER, ARTIFICIAL_PA, ARTIFICIAL_SMALL}:
        return _get_csv_paths(f"{net_type}/*--net-{net_name}.csv")
    elif net_type == FMRI74:
        return _get_csv_paths(f"fmri74/*.csv")
    elif net_type == ARXIV_NETSCIENCE_COAUTHORSHIP:
        if net_name == ARXIV_NETSCIENCE_COAUTHORSHIP:
            return [
                f for f in Path(f"{MLN_SP_DATA_PATH}/arxiv_netscience_coauthorship").rglob("**/*.csv") 
                if "math.oc" not in f.parts
            ]
        elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH:
            return _get_csv_paths(f"arxiv_netscience_coauthorship/math.oc/*.csv")
        raise AttributeError(f"Unknown network name: {net_name} for {net_type}")
    elif net_type == AUCS:
        return _get_csv_paths(f"small_real/*--net-aucs.csv")
    elif net_type == CANNES:
        _sp_not_implemented()
    elif net_type == CKM_PHYSICIANS:
        return _get_csv_paths("small_real/*--net-ckm_physicians.csv")
    elif net_type == EU_TRANSPORTATION:
        return _get_csv_paths(f"small_real/*--net-{net_name}.csv")
    elif net_type == L2_COURSE:
        return _get_csv_paths(f"l2_course_net/*--net-{net_name}.csv")
    elif net_type == LAZEGA:
        return _get_csv_paths("small_real/*--net-lazega.csv")
    elif net_type == TIMIK1Q2009:
        return _get_csv_paths("timik1q2009/**/*.csv")
    elif net_type == TOY_NETWORK:
        return _get_csv_paths("small_real/*--net-toy_network.csv")
    raise AttributeError(f"Unknown network: {net_type}")
