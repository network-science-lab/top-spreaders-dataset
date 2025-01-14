"""Calculate available centralities for selected multilayer networks."""

import numpy as np
import pandas as pd

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

from network_diffusion.mln.functions import remove_selfloop_edges
from network_diffusion.mln.mlnetwork import MultilayerNetwork, MLNetworkActor
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from nsl_data_utils.loaders.constants import (
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
    ARTIFICIAL_RANDOM,
    ARTIFICIAL_SMALL,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    ARXIV_NETSCIENCE_COAUTHORSHIP_MATH,
    AUCS,
    CANNES,
    CENTRALITY_FUNCTIONS,
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
from nsl_data_utils.loaders.net_loader import load_network, load_net_names


AVAILABLE_NETWORKS = [
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
    ARTIFICIAL_RANDOM,
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


def save_centralities(
    features_df: pd.DataFrame,
    network_name: str,
    instance_name: str,
    len_mln_networks: int,
) -> None:
    save_path = (
        MLN_CENTRALITIES_DATA_PATH / network_name
        if len_mln_networks > 1
        else MLN_CENTRALITIES_DATA_PATH
    )
    save_path.mkdir(exist_ok=True, parents=True)
    features_df.to_csv(save_path / f"{instance_name}.csv")


def calculate_centralities(mln_network: MultilayerNetwork) -> np.ndarray:
    mln_centralities: list[dict[MLNetworkActor, float]] = [
        centrality_function(mln_network)
        for centrality_function in tqdm(
            CENTRALITY_FUNCTIONS,
            desc="Centralities",
        )
    ]

    features_raw = {}
    for actor in mln_network.get_actors():
        actor_centralities = [
            mln_centrality[actor] if actor in mln_centrality else 0
            for mln_centrality in mln_centralities
        ]
        features_raw[actor.actor_id] = actor_centralities

    features_df = pd.DataFrame(features_raw).T
    features_df = features_df.set_axis([func.__name__ for func in CENTRALITY_FUNCTIONS], axis=1)

    return features_df


def _get_networks(net_type: str) -> dict[str, MultilayerNetwork]:
    net_names = load_net_names(net_type)
    return {
        net_name: load_network(
            net_type=net_type,
            net_name=net_name,
            as_tensor=False,
        )
        for net_name in net_names
    }


def calculate_and_save(network_name: str) -> None:
    mln_networks = _get_networks(network_name)
    for name, mln_network in tqdm(
        mln_networks.items(),
        desc="Network group",
    ):
        mln_network = remove_selfloop_edges(mln_network)
        features_df = calculate_centralities(mln_network=mln_network)
        save_centralities(
            features_df=features_df,
            network_name=network_name,
            instance_name=name,
            len_mln_networks=len(mln_networks),
        )


def parse_args(args=None) -> Namespace:
    """Parse CLI args according to the parser."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--network",
        type=str,
        default="all",
        choices=AVAILABLE_NETWORKS + ["all"],
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)

    return parser.parse_args(args)


def main(args: Namespace) -> None:
    processing_networks = (
        AVAILABLE_NETWORKS if args.network == "all" else [args.network]
    )

    process_map(
        calculate_and_save,
        processing_networks,
        max_workers=args.n_jobs,
        chunksize=args.chunksize,
    )


def cli() -> None:
    args = parse_args()
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
