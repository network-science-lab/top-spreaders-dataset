"""Generate multilayer networks with predefined network models."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import uunet.multinet as ml

from nsl_data_utils.misc.generators import MultilayerERGenerator, MultilayerPAGenerator
from tqdm.contrib.concurrent import process_map
from uunet._multinet import PyMLNetwork


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
    """Parse CLI args accorfding to the parser."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("min_actors", type=int)
    parser.add_argument("max_actors", type=int)
    parser.add_argument("-s", "--step-size", type=int, default=1)
    parser.add_argument("-o", "--output-dir", type=str, default="./generated_graphs/")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args(args)


def main(args: Namespace) -> None:
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
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
