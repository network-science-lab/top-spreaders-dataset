"""Calculate available centralities for selected multilayer networks."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

import numpy as np
from network_diffusion.mln import MLNetworkActor, MultilayerNetworkTorch
from network_diffusion.mln.functions import remove_selfloop_edges
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from nsl_data_utils.loaders.constants import (
    ARTIFICIAL_ER,
    ARTIFICIAL_PA,
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
from nsl_data_utils.loaders.net_loader import load_network
from tqdm.contrib.concurrent import process_map


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


def save_centralities(
    sorted_features: np.ndarray,
    network_name: str,
    instance_name: str,
    len_mln_networks: int,
) -> None:
    save_path = (
        MLN_CENTRALITIES_DATA_PATH / network_name
        if len_mln_networks > 1
        else MLN_CENTRALITIES_DATA_PATH
    )
    save_path.mkdir(
        exist_ok=True,
        parents=True,
    )

    with (save_path / f"{instance_name}.txt").open("w") as file:
        np.savetxt(
            fname=file,
            X=sorted_features,
        )


def calculate_centralities(
    mln_network: MultilayerNetwork,
    torch_network: MultilayerNetworkTorch,
) -> np.ndarray:
    mln_centralities: list[dict[MLNetworkActor, float]] = [
        centrality_function(mln_network) for centrality_function in CENTRALITY_FUNCTIONS
    ]

    features_raw = []
    actor_indices = []
    for actor in mln_network.get_actors():
        actor_indices.append(torch_network.actors_map[actor.actor_id])
        features_raw.append(
            [
                mln_centrality[actor] if actor in mln_centrality else 0
                for mln_centrality in mln_centralities
            ]
        )
    values = np.array(features_raw)

    actor_indices = np.array(actor_indices)
    sorted_features = values[actor_indices.argsort()]

    return sorted_features


def calculate_and_save(network_name: str) -> None:
    mln_networks = load_network(
        net_name=network_name,
        as_tensor=False,
    )
    for name, mln_network in mln_networks.items():
        mln_network = remove_selfloop_edges(mln_network)
        torch_network = MultilayerNetworkTorch.from_mln(mln_network)

        sorted_features = calculate_centralities(
            mln_network=mln_network,
            torch_network=torch_network,
        )

        save_centralities(
            sorted_features=sorted_features,
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
    processing_networks = AVAILABLE_NETWORKS if args.network == "all" else [args.network]

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
