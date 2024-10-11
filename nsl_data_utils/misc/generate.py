"""
Generate a multilayer network
"""
import abc

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import uunet.multinet as ml
from tqdm.contrib.concurrent import process_map
from uunet._multinet import PyMLNetwork, PyEvolutionModel



class MultilayerBaseGenerator(abc.ABC):
    """Abstract base class for multilayer network generators."""

    @abc.abstractmethod
    def __init__(
        self, nb_layers: int, nb_actors: int, nb_steps: int, random_state: int | None = None,
    ) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param random_state: RNG state to make the generation reproducible, defaults to None
        """
        self.nb_layers=nb_layers
        self.nb_actors=nb_actors
        self.nb_steps=nb_steps
        self.rng = np.random.default_rng(random_state)

    @abc.abstractmethod
    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of concrete network generators
        """
        pass

    def get_dependency(self) -> np.ndarray:
        """
        Create a dependency matrix for the network generator.

        :return: matrix shaped: [nb_layers, nb_layers] with 0s in diagonal and 1/nb_layers otherwise
        """
        dep = np.full(fill_value=(1 / self.nb_layers), shape=(self.nb_layers, self.nb_layers))
        np.fill_diagonal(dep, 0)
        return dep

    def __call__(self) -> PyMLNetwork:
        """
        Generate the network according to given parameters.

        :return: generated network as a PyEvolutionModel object
        """
        models = self.get_models()
        pr_internal = self.rng.uniform(0, 1, size=self.nb_layers)
        pr_external = np.ones_like(pr_internal) - pr_internal  # TODO: there is no noaction step now!
        dependency = self.get_dependency()
        return ml.grow(
            self.nb_actors,
            self.nb_steps,
            models,
            pr_internal.tolist(),
            pr_external.tolist(),
            dependency.tolist(),
        )


class MultilayerERGenerator(MultilayerBaseGenerator):
    """Class which generates multilayer network with Erdos-Renyi algorithm."""
    
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int, std_nodes: int, random_state: int | None = None) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param std_nodes: standard deviation of the number of nodes in each layer (expected value
            is a number of actors)
        :param random_state: RNG state to make the generation reproducible, defaults to None
        """
        super().__init__(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_steps,
            random_state=random_state
        )
        assert nb_actors - 2 * std_nodes > 0, "Number of actors shall be g.t. 2 * std_nodes"
        self.std_nodes=std_nodes

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of Erdos-Renyi generators
        """
        layer_sizes = self.rng.normal(
            loc=self.nb_actors - self.std_nodes, scale=self.std_nodes, size=self.nb_layers
        ).astype(int).clip(min=1, max=self.nb_actors)
        return [ml.evolution_er(n=lv) for lv in layer_sizes]


class MultilayerPAGenerator(MultilayerBaseGenerator):
    """Class which generates multilayer network with Preferential Attachment algorithm."""
    
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int, nb_hubs: int, random_state: int | None = None) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param std_nodes: standard deviation of the number of nodes in each layer (expected value
            is a number of actors)
        :param random_state: RNG state to make the generation reproducible, defaults to None
        """
        super().__init__(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_steps,
            random_state=random_state
        )
        self.nb_hubs = nb_hubs

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of Preferential Attachment generators
        """
        m0s = [self.nb_hubs] * self.nb_layers
        ms = m0s.copy()
        return [ml.evolution_pa(m0=m0, m=ms) for m0, ms in zip(m0s, ms)]


def generate_multilayer(
    layers: int,
    actors: int,
    steps: int,
    random_state: int | None = None,
) -> PyMLNetwork:
    rng = np.random.default_rng(random_state)

    models = [
        ml.evolution_er(n)
        for n in rng.integers(actors / 10, actors, size=layers)
    ]
    pr_internal = rng.uniform(0, 1, size=layers).tolist()
    pr_external = rng.uniform(0, 1, size=layers).tolist()

    dependency = np.full(
        fill_value=1 / layers, shape=(layers, layers)
    ).tolist()

    return ml.grow(actors, steps, models, pr_internal, pr_external, dependency)


def save_generate(
    layers: int,
    actors: int,
    steps: int,
    outfile: str,
    random_state: int | None = None,
) -> None:
    network = generate_multilayer(
        layers=layers, actors=actors, steps=steps, random_state=random_state
    )
    ml.write(network, str(outfile))


def parse_args(args=None) -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("min_actors", type=int)
    parser.add_argument("max_actors", type=int)

    parser.add_argument("-s", "--step-size", type=int, default=1)
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./generated_graphs/"
    )

    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=None)

    return parser.parse_args(args)


def main(args: Namespace):
    print(args)
    rng = np.random.default_rng(args.random_state)

    actors = np.arange(args.min_actors, args.max_actors, args.step_size, dtype=np.int32)
    layers = rng.integers(
        args.min_actors / 10,
        args.max_actors / 10,
        size=len(actors),
        dtype=np.int32,
    )
    steps = actors * 5

    random_states = rng.integers(
        np.iinfo(np.int32).max - 1,
        size=len(actors),
        dtype=np.int32,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    outfiles = (output_dir / f"network_{a}.mpx" for a in actors)

    process_map(
        save_generate,
        layers,
        actors,
        steps,
        outfiles,
        random_states,
        max_workers=args.n_jobs,
        chunksize=args.chunksize,
    )


def cli():
    import network_diffusion as nd
    args = parse_args(
        [
            "100",
            "1000",
            "--step-size", "1",
            "--output-dir", "/Users/michal/Development/infmax-trainer-icm-mln/dupa",
            "--n-jobs", "1",
            "--chunksize", "1",
            "--random-state", "1",
        ]
    )
    # main(args)
    net_uu = MultilayerERGenerator(nb_layers=3, nb_actors=30, nb_steps=40, std_nodes=14, random_state=43)()
    # ml.plot(n=net_uu)
    net_nx = ml.to_nx_dict(net_uu)
    net_nd = nd.MultilayerNetwork(layers=net_nx)
    nd.mln.functions.draw_mln(net_nd)
    # print(network)


if __name__ == "__main__":
    cli()
