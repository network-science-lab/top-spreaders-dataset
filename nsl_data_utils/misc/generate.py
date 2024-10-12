"""Generate multilayer networks with predefined network models."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import uunet.multinet as ml

from nsl_data_utils.misc.generators import MultilayerERGenerator, MultilayerPAGenerator
from tqdm.contrib.concurrent import process_map


def generate_and_save(model_type: str, actors: int, layers: int, steps: int, outfile: str) -> None:
    """Generate network and save it as an mpx file."""
    if model_type == "PA":
        nb_hubs = int(max(3, 0.05 * actors))
        generator = MultilayerPAGenerator(
            nb_layers=layers, nb_actors=actors, nb_steps=steps, nb_hubs=nb_hubs
        )
    elif model_type == "ER":
        std_nodes = int(0.1 * actors)
        generator = MultilayerERGenerator(
            nb_layers=layers, nb_actors=actors, nb_steps=steps, std_nodes=std_nodes
        )
    else:
        raise ValueError("Model type can be either PA or ER!")
    network = generator()
    ml.write(network, str(outfile))


def parse_args(args=None) -> Namespace:
    """Parse CLI args according to the parser."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("nb_networks", type=int)
    parser.add_argument("min_actors", type=int)
    parser.add_argument("max_actors", type=int)
    parser.add_argument("min_layers", type=int)
    parser.add_argument("max_layers", type=int)
    parser.add_argument("model_type", type=str, choices=["ER", "PA"])
    parser.add_argument("-o", "--output-dir", type=str, default="./generated_graphs/")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)
    return parser.parse_args(args)


def main(args: Namespace) -> None:
    """Main generation routine feed with CLI args."""
    actors = np.random.randint(args.min_actors, args.max_actors + 1, dtype=np.int32, size=args.nb_networks)
    layers = np.random.randint(args.min_layers, args.max_layers + 1, dtype=np.int32, size=args.nb_networks)
    steps = actors * 5
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    outfiles = (output_dir / f"network_{idx}.mpx" for idx, _ in enumerate(actors, 1))
    process_map(
        generate_and_save,
        [args.model_type] * len(actors),
        actors,
        layers,
        steps,
        outfiles,
        max_workers=args.n_jobs,
        chunksize=args.chunksize,
    )


def cli():
    args = parse_args(
        [
            "100",
            "100",
            "10000",
            "2",
            "5",
            "ER",
            "--output-dir", "/Users/michal/Development/infmax-trainer-icm-mln/krakrarkarka",
            "--n-jobs", "5",
            "--chunksize", "20"
        ]
    )
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
