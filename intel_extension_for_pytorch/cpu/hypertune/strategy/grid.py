import itertools
from .strategy import strategy_registry, TuneStrategy


@strategy_registry
class GridTuneStrategy(TuneStrategy):
    def __init__(self, conf):
        super().__init__(conf)
        self.combinations = itertools.product(
            *(self.hyperparam2searchspace[hp] for hp in self.hyperparams)
        )

    def next_tune_cfg(self):
        for comb in self.combinations:
            tune_cfg = dict(zip(self.hyperparams, comb))
            yield tune_cfg
        return
