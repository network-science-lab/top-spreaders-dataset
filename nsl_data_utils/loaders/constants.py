"""A script with strings used across module in form of variables."""

from pathlib import Path

from network_diffusion.mln.functions import (
    betweenness,
    closeness,
    core_number,
    degree,
    neighbourhood_size,
)
from nsl_data_utils.loaders import voterank

MLN_RAW_DATA_PATH = (
    Path(__file__).parent.parent.parent / "nsl_data_sources/raw/multi_layer_networks"
)
MLN_SP_DATA_PATH = (
    Path(__file__).parent.parent.parent
    / "nsl_data_sources/spreading_potentials/multi_layer_networks"
)
MLN_CENTRALITIES_DATA_PATH = (
    Path(__file__).parent.parent.parent
    / "nsl_data_sources/centralities/multi_layer_networks"
)

# columns in the dataset
ACTOR = "actor"
EXPOSED = "exposed"
NETWORK = "network"
NOT_EXPOSED = "not_exposed"
P = "p"
PEAK_INFECTED = "peak_infected"
PEAK_ITERATION = "peak_iteration"
PROTOCOL = "protocol"
SIMULATION_LENGTH = "simulation_length"

# allowed protocols
OR = "OR"
AND = "AND"

# network names
ARTIFICIAL_ER = "artificial_er"  # a bunch of nets
ARTIFICIAL_PA = "artificial_pa"  # a bunch of nets
ARTIFICIAL_SMALL = "artificial_small"  # a bunch of nets
ARXIV_NETSCIENCE_COAUTHORSHIP = "arxiv_netscience_coauthorship"
ARXIV_NETSCIENCE_COAUTHORSHIP_MATH = "arxiv_netscience_coauthorship_math.oc"
AUCS = "aucs"
CANNES = "cannes"
CKM_PHYSICIANS = "ckm_physicians"
EU_TRANSPORTATION = "eu_transportation"
EU_TRANSPORT_KLM = "eu_transport_klm"
FMRI74 = "fmri74"
L2_COURSE = "l2_course"
L2_COURSE_NET_1 = "l2_course_net_1"
L2_COURSE_NET_2 = "l2_course_net_2"
L2_COURSE_NET_3 = "l2_course_net_3"
LAZEGA = "lazega"
TIMIK1Q2009 = "timik1q2009"
TOY_NETWORK = "toy_network"

CENTRALITY_FUNCTIONS = [
    degree,
    betweenness,
    closeness,
    core_number,  # related with k-shell-mln
    neighbourhood_size,
    voterank,
]
