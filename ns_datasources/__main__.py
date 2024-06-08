from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    return parser.parse_args()


def main(args: Namespace):
    print(args)


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
