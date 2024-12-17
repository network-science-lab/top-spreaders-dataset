"""Loader for the precomputed centralities."""

from pathlib import Path
import pandas as pd
from nsl_data_utils.loaders.constants import (
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
    ARTIFICIAL_SMALL,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    AUCS,
    CKM_PHYSICIANS,
    EU_TRANSPORTATION,
    FMRI74,
    L2_COURSE,
    LAZEGA,
    MLN_CENTRALITIES_DATA_PATH,
    TIMIK1Q2009,
    TOY_NETWORK,
)


AVAILABLE_NETWORK_TYPES = [
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
    ARTIFICIAL_SMALL,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    AUCS,
    CKM_PHYSICIANS,
    EU_TRANSPORTATION,
    FMRI74,
    L2_COURSE,
    LAZEGA,
    TIMIK1Q2009,
    TOY_NETWORK,
]


def load_centralities(csv_path: Path) -> pd.DataFrame:
    centralities_df = pd.read_csv(csv_path, index_col=0)
    centralities_df.index = centralities_df.index.astype(str)
    return centralities_df


def load_centralities_path(network_type: str, network_name: str) -> Path:
    if network_type not in AVAILABLE_NETWORK_TYPES:
        raise NotImplementedError(f"Centralities for {network_name} are not available yet.")
    save_path = MLN_CENTRALITIES_DATA_PATH / network_type
    if network_type in {ARTIFICIAL_ER, ARTIFICIAL_PA, ARTIFICIAL_SMALL}:
        return_p =  save_path / f"{network_name}.csv"
    else:
        return_p = save_path.parent / f"{network_name}.csv"
    assert return_p.exists()
    return return_p
