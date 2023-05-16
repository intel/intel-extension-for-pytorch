# reference: https://github.com/intel/neural-compressor/blob/\
# 15477100cef756e430c8ef8ef79729f0c80c8ce6/neural_compressor/strategy/strategy.py
import os
from abc import abstractmethod
import csv
from collections import OrderedDict
import click
from ..objective import MultiObjective

STRATEGIES = {}


def strategy_registry(cls):
    assert cls.__name__.endswith(
        "TuneStrategy"
    ), "The name of subclass of TuneStrategy should end with 'TuneStrategy' substring."
    if cls.__name__[: -len("TuneStrategy")].lower() in STRATEGIES:
        raise ValueError("Cannot have two strategies with the same name")
    STRATEGIES[cls.__name__[: -len("TuneStrategy")].lower()] = cls
    return cls


class TuneStrategy(object):
    def __init__(self, conf):
        self.conf = conf.execution_conf
        self.program = conf.program
        self.program_args = conf.program_args
        self.usr_objectives = conf.usr_objectives

        self.max_trials = conf.execution_conf.tuning.max_trials

        # hyperparams #
        self.hyperparam2searchspace = OrderedDict()
        for k in self.conf.hyperparams:
            for hp in self.conf.hyperparams[k]["hp"]:
                self.hyperparam2searchspace[hp] = self.conf.hyperparams[k][hp]
        self.hyperparams = list(self.hyperparam2searchspace.keys())
        tune_launcher = "launcher" in self.conf.hyperparams

        # objective #
        self.multiobjective = MultiObjective(
            self.program, self.program_args, tune_launcher
        )

        # output #
        output_name = "record.csv"
        log_name = os.path.join(self.conf.output_dir, output_name)
        csvfile = open(log_name, "w", newline="")
        self.tune_result_record = csv.writer(csvfile, delimiter=",")
        self.tune_result_record.writerow(
            list(self.hyperparam2searchspace.keys())
            + [objective["name"] for objective in self.usr_objectives]
        )

        self.best_tune_result = None
        self.best_tune_cfg = None

    @abstractmethod
    def next_tune_cfg(self):
        raise NotImplementedError

    def traverse(self):
        click.secho("Starting hypertuning...", fg="green")
        trials_count = 0

        for tune_cfg in self.next_tune_cfg():
            trials_count += 1

            click.secho("\nTune ", fg="green", nl=False)
            click.secho("{trials_count}", fg="blue", nl=False)

            click.secho("\nCurrent configuration is: ", fg="green", nl=False)
            click.secho("{tune_cfg}", fg="blue")

            curr_tune_result = self.multiobjective.evaluate(tune_cfg)

            self._update_best_tune_result(curr_tune_result, tune_cfg)
            self._record_tune_result(curr_tune_result, tune_cfg)

            need_stop = self._stop(trials_count)

            if need_stop:
                # case 1: accuracy goal is met
                # case 2: timeout reached (objective goal not met)
                self._print_best_result()
                return

        # finished traversal
        # case 3: finished traversal (objective goal not met)
        click.secho(
            "\nFinished traversing the entire search space, but didn't find configuration meeting the objective goal",
            fg="red",
        )
        self._print_best_result()
        return

    def _compare(self, higher_is_better, src, dst):
        if higher_is_better:
            return src > dst
        else:
            return src < dst

    def _update_best_tune_result(self, curr_tune_result, curr_tune_cfg):
        if self.best_tune_result is None and self.best_tune_cfg is None:
            # initial baseline
            self.best_tune_result = curr_tune_result
            self.best_tune_cfg = curr_tune_cfg
        else:
            # multi objective
            if all(
                [
                    self._compare(higher_is_better, curr_val, best_val)
                    for higher_is_better, curr_val, best_val in zip(
                        [
                            objective["higher_is_better"]
                            for objective in self.usr_objectives
                        ],
                        curr_tune_result,
                        self.best_tune_result,
                    )
                ]
            ):
                self.best_tune_result = curr_tune_result
                self.best_tune_cfg = curr_tune_cfg

    def _record_tune_result(self, curr_tune_result, curr_tune_cfg):
        for objective, val in zip(self.usr_objectives, curr_tune_result):
            click.secho("{objective['name']}: {val}", fg="blue")

        click.secho("Best configuration is: ", fg="green", nl=False)
        click.secho("{self.best_tune_cfg}", fg="blue")
        for objective, val in zip(self.usr_objectives, self.best_tune_result):
            click.secho("{objective['name']}: {val}", fg="blue")

        curr_tune_cfg_val = list(_ for _ in curr_tune_cfg.values())
        self.tune_result_record.writerow(curr_tune_cfg_val + curr_tune_result)

    def _stop(self, trials_count):
        if all(
            [
                self._compare(higher_is_better, best_val, target_val)
                for higher_is_better, best_val, target_val in zip(
                    [
                        objective["higher_is_better"]
                        for objective in self.usr_objectives
                    ],
                    self.best_tune_result,
                    [objective["target_val"] for objective in self.usr_objectives],
                )
            ]
        ):
            click.secho("\nFound configuration meeting the target values.", fg="red")
            return True
        elif trials_count == self.max_trials:
            click.secho(
                "\nMax trials is reached, but didn't find configuration meeting the objective goal.",
                fg="red",
            )
            return True
        return False

    def _print_best_result(self):
        click.secho("Best configuration found is: ", fg="green", nl=False)
        click.secho("{self.best_tune_cfg}", fg="blue")
        for objective, val in zip(self.usr_objectives, self.best_tune_result):
            click.secho("{objective['name']}: {val}", fg="blue")
