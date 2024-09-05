"""A script with strings used across module in form of variables."""

from pathlib import Path

MLN_RAW_DATA_PATH = Path(__file__).parent.parent.parent / "ns-data-sources/raw/multi_layer_networks"
MLN_SP_DATA_PATH = Path(__file__).parent.parent.parent / "ns-data-sources/spreading_potentials/multi_layer_networks"

# columns in the dataset
ACTOR = "actor"
EXPOSED="exposed"
NETWORK = "network"
NOT_EXPOSED="not_exposed"
P = "p"
PEAK_INFECTED="peak_infected"
PEAK_ITERATION="peak_iteration"
PROTOCOL = "protocol"
SIMULATION_LENGTH="simulation_length"

# network names
ARXIV_NETSCIENCE_COAUTHORSHIP = "arxiv_netscience_coauthorship"
ARXIV_NETSCIENCE_COAUTHORSHIP_MATH = "arxiv_netscience_coauthorship_math.oc"
AUCS = "aucs"
CANNES = "cannes"
CKM_PHYSICIANS = "ckm_physicians"
EU_TRANSPORTATION = "eu_transportation"
EU_TRANSPORT_KLM = "eu_transport_klm"
ER1 = "er1"
ER2 = "er2"
ER3 = "er3"
ER5 = "er5"
FMRI74 = "fmri74"
LAZEGA = "lazega"
SF1 = "sf1"
SF2 = "sf2"
SF3 = "sf3"
SF5 = "sf5"
TIMIK1Q2009 = "timik1q2009"
TOY_NETWORK = "toy_network"
