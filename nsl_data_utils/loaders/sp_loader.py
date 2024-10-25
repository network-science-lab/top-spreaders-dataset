"""A loader for the dataset of spreading potentials."""

import re
from pathlib import Path
from typing import Any

import pandas as pd
from nsl_data_utils.loaders.constants import (
    ACTOR,
    EXPOSED,
    MLN_SP_DATA_PATH,
    NETWORK,
    NOT_EXPOSED,
    P,
    PEAK_INFECTED,
    PEAK_ITERATION,
    PROTOCOL,
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
    EU_TRANSPORT_KLM,
    FMRI74,
    L2_COURSE_NET_1,
    L2_COURSE_NET_2,
    L2_COURSE_NET_3,
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


def _get_sp(csv_paths: list[str]) -> pd.DataFrame:
    """Read spreading potentials stored in files indicated by given regex."""
    raw_csvs = []
    for _, file_path in enumerate(csv_paths):
        raw_csvs.append(read_csv(file_path))
    sp = pd.concat(raw_csvs, axis=0, ignore_index=True)
    assert len(sp["network"].unique()) == 1
    sp[[SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION, P]] = sp[
        [SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION, P]
    ].apply(pd.to_numeric)
    return sp


def _get_csv_paths(csv_regex: str) -> list[Path]:
    return list(Path(MLN_SP_DATA_PATH).glob(csv_regex))


def _sort_csv_paths(csv_regex: str):
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


def _get_sp_bulk(net_group_name: str) -> dict[str, pd.DataFrame]:
    """Bulk loader of SP for given network group."""
    sp_dict = {}
    for net_name, csv_paths in _sort_csv_paths(f"{net_group_name}/*.csv").items():
        sp_dict[net_name] = _get_sp(csv_paths)
    return sp_dict


def _sp_not_implemented():
    raise NotImplementedError(f"Spreading potentials have been not prepared yet!")


def load_sp(net_name: str) -> dict[str, pd.DataFrame]:
    """Load spreading potentials dataset for given network."""
    if net_name == ARTIFICIAL_ER:
        return _get_sp_bulk(ARTIFICIAL_ER)
    elif net_name == ARTIFICIAL_PA:
        return _get_sp_bulk(ARTIFICIAL_PA)
    elif net_name == ARTIFICIAL_SMALL:
        return _get_sp_bulk("small_artificial")
    elif net_name == FMRI74:
        return _get_sp(_get_csv_paths(f"{MLN_SP_DATA_PATH}/fmri74/*.csv"))
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP:
        csv_paths = [
            f for f in Path(f"{MLN_SP_DATA_PATH}/arxiv_netscience_coauthorship").rglob('**/*.csv') 
            if 'math.oc' not in f.parts
        ]
        return {net_name: _get_sp(csv_paths)}
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH:
        return {net_name: _get_sp(_get_csv_paths(f"arxiv_netscience_coauthorship/math.oc/*.csv"))}
    elif net_name == AUCS:
        return {net_name: _get_sp(_get_csv_paths(f"small_real/*--net-aucs.csv"))}
    elif net_name == CANNES:
        _sp_not_implemented()
    elif net_name == CKM_PHYSICIANS:
        return {net_name: _get_sp(_get_csv_paths("small_real/*--net-ckm_physicians.csv"))}
    elif net_name == EU_TRANSPORTATION:
        return {net_name: _get_sp(_get_csv_paths("small_real/*--net-eu_transportation.csv"))}
    elif net_name == EU_TRANSPORT_KLM:
        return {net_name: _get_sp(_get_csv_paths("small_real/*--net-eu_transport_klm.csv"))}
    elif net_name == L2_COURSE_NET_1:
        return {net_name: _get_sp(_get_csv_paths("l2_course_net/*--net-l2_course_net_1.csv"))}
    elif net_name == L2_COURSE_NET_2:
        return {net_name: _get_sp(_get_csv_paths("l2_course_net/*--net-l2_course_net_2.csv"))}
    elif net_name == L2_COURSE_NET_3:
        return {net_name: _get_sp(_get_csv_paths("l2_course_net/*--net-l2_course_net_3.csv"))}
    elif net_name == LAZEGA:
        return {net_name: _get_sp(_get_csv_paths("small_real/*--net-lazega.csv"))}
    elif net_name == TIMIK1Q2009:
        return {net_name: _get_sp(_get_csv_paths("timik1q2009/**/*.csv"))}
    elif net_name == TOY_NETWORK:
        return {net_name: _get_sp(_get_csv_paths("small_real/*--net-toy_network.csv"))}
    raise AttributeError(f"Unknown network: {net_name}")


def get_gt_data(sp_raw: pd.DataFrame, protocol: str, p: float | None, budget: int) -> list[Any]:
    """
    Get actors that performed the best in given spreading contitions.

    :param sp_raw: DataFrame loaded with `load_sp` with raw spreading potentials for a given network
    :param protocol: protocol of the multilayer ICM
    :param p: probability of the multilayer ICM, if not provided probability will be discarded in
        the process of selecting top-k actors
    :param budget: top-k actors to return
    :return: IDs of actors that performed the best in given contidions
    """
    if p:
        sp_mean  = sp_raw.groupby(by=[NETWORK, PROTOCOL, ACTOR, P]).mean().reset_index()
        sp_mean = sp_mean[(sp_mean[PROTOCOL] == protocol) & (sp_mean[P] == p)]
    else:
        sp_mean  = sp_raw.groupby(by=[NETWORK, PROTOCOL, ACTOR]).mean().reset_index()
        sp_mean = sp_mean[sp_mean[PROTOCOL] == protocol]
    sp_mean = sp_mean.sort_values(
        [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
        ascending=[False, True, True, False]
    )
    return sp_mean.iloc[:budget][ACTOR].tolist()



if __name__ == "__main__":
    a = load_sp(ARTIFICIAL_SMALL)
    print(a.keys())
    print(get_gt_data(a["sf1"], "OR", 0.5, 10))
