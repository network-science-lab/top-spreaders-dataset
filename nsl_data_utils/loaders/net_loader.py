"""A loader for multilayer networks stored in the dataset."""

from functools import wraps
from pathlib import Path
from typing import Callable

import pandas as pd
import network_diffusion as nd
import networkx as nx
import torch
from bidict import bidict
from tqdm import tqdm

from nsl_data_utils.loaders.constants import (
    MLN_RAW_DATA_PATH,
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
from nsl_data_utils.loaders.fmri74 import read_fmri74


def _network_from_pandas(path):
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in df["layer"].unique()}
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_dict.keys()), network_list=list(net_dict.values())
    )


def return_some_layers(get_network_func: Callable) -> Callable:
    """Decorator for network loader to filter out a multilayer network to return only one layer."""
    @wraps(get_network_func)
    def wrapper(layer_slice = None):
        net = get_network_func()
        if not layer_slice:
            return net
        l_graphs = [net.layers[layer] for layer in layer_slice]
        return nd.MultilayerNetwork.from_nx_layers(l_graphs, layer_slice)
    return wrapper


def get_aucs_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_real/aucs.mpx")


def get_ckm_physicians_network():
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/CKM-Physicians-Innovation_4NoNature.edges"
    )


@return_some_layers
def get_eu_transportation_network():
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/EUAirTransportation_multiplex_4NoNature.edges"
    )


def get_lazega_network():
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )


def get_ddm_network(layernames_path, edgelist_path, weighted, digraph):
    # read mapping of layer IDs to their names
    with open(layernames_path, encoding="utf-8") as file:
        layer_names = file.readlines()
    layer_names = [ln.rstrip('\n').split(" ") for ln in layer_names]
    layer_names = {ln[0]: ln[1] for ln in layer_names}
    
    # read the edgelist and create containers for the layers
    df = pd.read_csv(
        edgelist_path,
        names=["layer_id", "node_1", "node_2", "weight"],
        sep=" "
    )
    net_ids_dict = {
        l_name: nx.DiGraph() if digraph else nx.Graph()
        for l_name in list(df["layer_id"].unique())
    }

    # populate network with edges
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        if weighted:
            attrs = {"weight": row["weight"]}
        else:
            attrs = {}
        net_ids_dict[row["layer_id"]].add_edge(row["node_1"], row["node_2"], **attrs)
    
    # rename layers
    net_names_dict = {
        layer_names[str(layer_id)]: layer_graph
        for layer_id, layer_graph in net_ids_dict.items()
    }

    # create multilater network from edges
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_names_dict.keys()), network_list=list(net_names_dict.values())
    )


@return_some_layers
def get_arxiv_network():
    root_path = Path(f"{MLN_RAW_DATA_PATH}/arxiv_netscience_coauthorship/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "arxiv_netscience_layers.txt",
        edgelist_path=root_path / "arxiv_netscience_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_cannes_network():
    root_path = Path(f"{MLN_RAW_DATA_PATH}/cannes_2013_social/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "Cannes2013_layers.txt",
        edgelist_path=root_path / "Cannes2013_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_timik1q2009_network():
    layer_graphs = []
    layer_names = []
    for i in Path(f"{MLN_RAW_DATA_PATH}/timik1q2009").glob("*.csv"):
        layer_names.append(i.stem)
        layer_graphs.append(nx.from_pandas_edgelist(pd.read_csv(i)))
    return nd.MultilayerNetwork.from_nx_layers(layer_graphs, layer_names)


def get_artificial_nets(dir_name: str) -> dict[str, nd.MultilayerNetwork]:
    nets_dict = {}  # TODO: this function is super slow
    paths = list(Path(f"{MLN_RAW_DATA_PATH}/{dir_name}").glob("*.mpx"))
    for path in (p_bar := tqdm(paths)):
        p_bar.set_description_str(f"Loading network: {path.stem}")
        nets_dict[path.stem] = nd.MultilayerNetwork.from_mpx(str(path))
    return nets_dict


def convert_to_torch(load_networks_func: Callable) -> Callable:
    """Decorate loader function so that it can convert the network on the fly to the tensor repr."""
    @wraps(load_networks_func)
    def wrapper(
        *args, as_tensor: bool, **kwargs
    ) -> dict[str, nd.MultilayerNetwork] | dict[str, nd.MultilayerNetworkTorch]:
        net_nd_dict = load_networks_func(*args, **kwargs)
        if as_tensor:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            net_pt_dict = {}
            for net_name, net_nd in net_nd_dict.items():
                net_pt = nd.MultilayerNetworkTorch.from_mln(net_nd, device=device)
                net_pt.actors_map = bidict(
                    {str(a_id): a_idx for a_id, a_idx in net_pt.actors_map.items()}
                )
                net_pt_dict[net_name] = net_pt
            return net_pt_dict
        return net_nd_dict
    return wrapper


@convert_to_torch
def load_network(net_name: str) -> dict[str, nd.MultilayerNetwork]:
    if net_name == ARTIFICIAL_ER:
        return get_artificial_nets("artificial_er")
    elif net_name == ARTIFICIAL_PA:
        return get_artificial_nets("artificial_pa")
    elif net_name == ARTIFICIAL_SMALL:
        nets_dict = get_artificial_nets("artificial_small")
        nets_dict["er1"] = nd.MultilayerNetwork.from_nx_layers([nets_dict["er5"]["l2"]], ["l2"])
        nets_dict["sf1"] = nd.MultilayerNetwork.from_nx_layers([nets_dict["sf5"]["l3"]], ["l3"])
        return nets_dict
    elif net_name == FMRI74:
        return {net_name: read_fmri74(f"{MLN_RAW_DATA_PATH}/CONTROL_fmt", True, 0.5)}
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP:
        return {net_name: get_arxiv_network()}
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH:
        return {net_name: get_arxiv_network(["math.OC"])}
    elif net_name == AUCS:
        return {net_name: get_aucs_network()}
    elif net_name == CANNES:
        return {net_name: get_cannes_network()}
    elif net_name == CKM_PHYSICIANS:
        return {net_name: get_ckm_physicians_network()}
    elif net_name == EU_TRANSPORTATION:
        return {net_name: get_eu_transportation_network()}
    elif net_name == EU_TRANSPORT_KLM:
        return {net_name: get_eu_transportation_network(["KLM"])}
    elif net_name == L2_COURSE_NET_1:
        net = nd.tpn.get_l2_course_net(node_features=True, edge_features=True, directed=False)
        return {L2_COURSE_NET_1: net.snaps[0]}
    elif net_name == L2_COURSE_NET_2:
        net = nd.tpn.get_l2_course_net(node_features=True, edge_features=True, directed=False)
        return {L2_COURSE_NET_2: net.snaps[1]}
    elif net_name == L2_COURSE_NET_3:
        net = nd.tpn.get_l2_course_net(node_features=True, edge_features=True, directed=False)
        return {L2_COURSE_NET_3: net.snaps[2]}
    elif net_name == LAZEGA:
        return {net_name: get_lazega_network()}
    elif net_name == TIMIK1Q2009:
        return {net_name: get_timik1q2009_network()}
    elif net_name == TOY_NETWORK:
        return {net_name: nd.mln.functions.get_toy_network_piotr()}
    raise AttributeError(f"Unknown network: {net_name}")
