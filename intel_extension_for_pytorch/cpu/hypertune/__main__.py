from argparse import ArgumentParser, REMAINDER
import click
from .conf.config import Conf
from .strategy.strategy import STRATEGIES


class Hypertune:
    def __init__(self, args):
        self.conf = Conf(args.conf_file, args.program, args.program_args)

        click.secho("Execution conf: ", fg="green", nl=False)
        click.secho(f"{self.conf.execution_conf}", fg="blue")

        click.secho("Tuning for: ", fg="green", nl=False)
        click.secho(f"{self.conf.usr_objectives}\n", fg="blue")

        self.strategy = STRATEGIES[self.conf.execution_conf.tuning.strategy](self.conf)

    def tune(self):
        self.strategy.traverse()


def parse_args():
    parser = ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--conf-file", "--conf_file", help="Configuration file for hypertuning"
    )
    parser.add_argument(
        "program",
        type=str,
        help="The full path to the proram/script to be launched. followed by all the arguments for the script",
    )
    parser.add_argument("program_args", nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()
    hypertune = Hypertune(args)
    hypertune.tune()


if __name__ == "__main__":
    main()
