import json
import logging
import os

import torch
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("qm_retrieval")
class QMRetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        accuracy = model.compute_accuracy(data_loader, task_cfg=self.cfg)

        if is_main_process():
            eval_result = self._report_metrics(accuracy)
            logging.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(accuracy):
        eval_result = {
           "accuracy": accuracy,
           "agg_metrics": accuracy
        }

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result
