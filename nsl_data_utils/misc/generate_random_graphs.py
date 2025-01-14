import random
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import networkx as nx
from nsl_data_utils.loaders.constants import MLN_RAW_DATA_PATH
from uunet import multinet


def generate_multilayer_graph(
    n: int, num_layers: int, avg_degree: int
) -> dict[int, nx.Graph]:
    """
    Creates random graph with given parameters
    """
    multilayer_graph = {}
    p = avg_degree / (n - 1)
    p = p / num_layers

    for layer in range(num_layers):
        G = nx.erdos_renyi_graph(n, p)
        multilayer_graph[layer] = G

    return multilayer_graph


def random_edge_rewiring_multilayer(
    multilayer_graph: dict[int, nx.Graph], iterations: int
) -> dict[int, nx.Graph]:
    """
    Random edge swapping on every layer
    """
    for _, graph in multilayer_graph.items():
        for _ in range(iterations):
            edges = list(graph.edges())

            edge1, edge2 = random.sample(edges, 2)
            u1, v1 = edge1
            u2, v2 = edge2

            if (
                u1 != u2
                and v1 != v2
                and not graph.has_edge(u1, v2)
                and not graph.has_edge(u2, v1)
            ):
                graph.remove_edge(u1, v1)
                graph.remove_edge(u2, v2)
                graph.add_edge(u1, v2)
                graph.add_edge(u2, v1)

    return multilayer_graph


def degree_preserving_randomization_multilayer(
    multilayer_graph: dict[int, nx.Graph],
    iterations: int,
) -> dict[int, nx.Graph]:
    """
    Edge swapping with preserved degrees on layers
    """
    for _, graph in multilayer_graph.items():
        i = 0
        while i < iterations:
            degree_sequence = dict(graph.degree())
            edges = list(graph.edges())
            edge1, edge2 = random.sample(edges, 2)
            u1, v1 = edge1
            u2, v2 = edge2

            if (
                u1 != u2
                and v1 != v2
                and not graph.has_edge(u1, v2)
                and not graph.has_edge(u2, v1)
            ):
                graph.remove_edge(u1, v1)
                graph.remove_edge(u2, v2)
                graph.add_edge(u1, v2)
                graph.add_edge(u2, v1)

                if (
                    degree_sequence[u1] == graph.degree(u1)
                    and degree_sequence[v1] == graph.degree(v1)
                    and degree_sequence[u2] == graph.degree(u2)
                    and degree_sequence[v2] == graph.degree(v2)
                ):
                    i += 1

                else:
                    try:
                        graph.remove_edge(u1, v2)
                        graph.remove_edge(u2, v1)
                        graph.add_edge(u1, v1)
                        graph.add_edge(u2, v2)
                    except:
                        continue

    return multilayer_graph


def save(network: dict[int, nx.Graph], idx: int) -> None:
    mnet = multinet.empty()
    save_path = MLN_RAW_DATA_PATH / "artificial_random"
    save_path.mkdir(exist_ok=True)

    for layer_id, layer in network.items():
        multinet.add_nx_layer(n=mnet, g=layer, name=f"l_{layer_id}")

    multinet.write(mnet, str(save_path / f"network_{idx}.mpx"))


def parse_args(args=None) -> Namespace:
    """Parse CLI args according to the parser."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-graphs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--avg-degree", type=list[int], default=[10, 20, 30, 40, 50])

    return parser.parse_args(args)


def main(args: Namespace) -> None:
    idx = 0

    for i in range(args.n_graphs):
        n = random.randint(100, 200)  # Nodes number
        num_layers = random.randint(2, 5)  # Number of layers
        avg_degree = args.avg_degree[i // len(args.avg_degree)]  # AVG node degree

        multilayer_graph1 = generate_multilayer_graph(
            n=n,
            num_layers=num_layers,
            avg_degree=avg_degree,
        )
        multilayer_random = random_edge_rewiring_multilayer(
            multilayer_graph=multilayer_graph1,
            iterations=args.iterations,
        )

        idx += 1
        save(
            network=multilayer_random,
            idx=idx,
        )

        multilayer_graph2 = generate_multilayer_graph(
            n=n,
            num_layers=num_layers,
            avg_degree=avg_degree,
        )
        multilayer_degree_preserving = degree_preserving_randomization_multilayer(
            multilayer_graph=multilayer_graph2,
            iterations=args.iterations,
        )

        idx += 1
        save(
            network=multilayer_degree_preserving,
            idx=idx,
        )


def cli() -> None:
    args = parse_args()
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
