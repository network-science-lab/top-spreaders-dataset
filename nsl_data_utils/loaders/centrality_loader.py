"""Loader for the precomputed centralities."""

import pandas as pd
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


def load_centralities(
    network_name: str,
    network_type: str,
) -> pd.DataFrame:
    if network_type not in AVAILABLE_NETWORKS:
        raise NotImplementedError(f"Centralities for {network_name} are not available yet.")
    save_path = MLN_CENTRALITIES_DATA_PATH / network_type
    if network_type != network_name:
        centralities_df = pd.read_csv(save_path / f"{network_name}.csv", index_col=0)
    else:
        centralities_df = pd.read_csv(save_path.parent / f"{save_path.stem}.csv", index_col=0)
    centralities_df.index = centralities_df.index.astype(str)
    return centralities_df
