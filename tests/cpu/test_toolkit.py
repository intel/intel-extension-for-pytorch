import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
import sklearn.metrics
import numpy as np

class ToolkitTester(TestCase) :

    def test_multi_thread_sklearn_metric_eval_roc_auc_score(self):
        targets = np.random.randint(0, 2, size=10)
        scores = torch.rand(10)
        roc_auc_st = sklearn.metrics.roc_auc_score(targets, scores.numpy())
        accuracy_st = sklearn.metrics.accuracy_score(y_true=targets, y_pred=np.round(scores.numpy()))
        roc_auc_mt, _, accuracy_mt = ipex._C.roc_auc_score_all(torch.Tensor(targets), scores)
        # For code coverage
        roc_auc_mt_2, _, _ = ipex._C.roc_auc_score(torch.Tensor(targets), scores)
        self.assertEqual(roc_auc_st, roc_auc_mt)
        self.assertEqual(roc_auc_st, roc_auc_mt_2)
        self.assertEqual(accuracy_st, accuracy_mt)
