from pathlib import Path

import numpy as np
from nsl_data_utils.loaders.constants import (
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
    ARTIFICIAL_SMALL,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    ARXIV_NETSCIENCE_COAUTHORSHIP_MATH,
    AUCS,
    CANNES,
    CKM_PHYSICIANS,
    EU_TRANSPORT_KLM,
    EU_TRANSPORTATION,
    FMRI74,
    L2_COURSE_NET_1,
    L2_COURSE_NET_2,
    L2_COURSE_NET_3,
    LAZEGA,
    MLN_CENTRALITIES_DATA_PATH,
    TIMIK1Q2009,
    TOY_NETWORK,
)

AVAILABLE_NETWORKS = [
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
]


def _load_file(path: Path) -> np.ndarray:
    with path.open("r") as file:
        sorted_features = np.loadtxt(
            fname=file,
            dtype=float,
        )

    return sorted_features


def load_centralities(
    network_name: str,
    network_type: int | None = None,
) -> np.ndarray:
    if network_name not in AVAILABLE_NETWORKS:
        raise NotImplementedError(
            f"Centralities for {network_name} are not available yet."
        )

    save_path = MLN_CENTRALITIES_DATA_PATH / network_name
    if network_type:
        return _load_file(save_path / f"network_{network_type}.txt")
    else:
        return _load_file(save_path.parent / f"{save_path.stem}.txt")
