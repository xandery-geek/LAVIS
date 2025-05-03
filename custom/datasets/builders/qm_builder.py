from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from custom.datasets.datasets.qm_dataset import QMRetrievalDataset, QMRetrievalEvalDataset

from lavis.common.registry import registry


@registry.register_builder("qm_retrieval")
class QMRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = QMRetrievalDataset
    eval_dataset_cls = QMRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/qm_retrieval.yaml"}

    def _download_data(self):
        # This dataset does not require downloading data
        return
