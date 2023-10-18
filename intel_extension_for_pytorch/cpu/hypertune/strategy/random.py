import itertools
import numpy as np
from .strategy import strategy_registry, TuneStrategy


@strategy_registry
class RandomTuneStrategy(TuneStrategy):
    def __init__(self, conf):
        super().__init__(conf)
        self.combinations = list(
            itertools.product(
                *(self.hyperparam2searchspace[hp] for hp in self.hyperparams)
            )
        )
        self.total_idx = set(i for i in range(len(self.combinations)))
        self.record_idx = set()

    def next_tune_cfg(self):
        while len(self.total_idx) > 0:
            idx = np.random.choice(list(self.total_idx))
            self.record_idx.add(idx)
            self.total_idx = self.total_idx - self.record_idx

            cfg = self.combinations[idx]
            tune_cfg = dict(zip(self.hyperparams, cfg))
            yield tune_cfg
        return
