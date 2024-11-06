"""A stub for data loaders."""

from network_diffusion.mln import MLNetworkActor
from network_diffusion.mln.functions import voterank_actorwise
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def voterank(net: MultilayerNetwork) -> dict[MLNetworkActor, int]:
    actors = voterank_actorwise(net)
    return {actor: idx for idx, actor in enumerate(actors[::-1])}
