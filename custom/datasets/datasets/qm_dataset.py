import os
import json
import random

from collections import OrderedDict
from PIL import Image
from lavis.datasets.datasets.base_dataset import BaseDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class QMRetrievalDataset(BaseDataset, __DisplMixin):
    _INAT21_MAPPING_FILE = "inaturalist_id2name.json"

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.inat21_dir = self.vis_root.inat21
        self.gldv2_dir = self.vis_root.gldv2
        self.kb_image_dir = self.vis_root.knowledge_base

        self.inat21_map = json.load(open(os.path.join(self.vis_root.inat21, self._INAT21_MAPPING_FILE)))
    
    def _get_image_path(self, dataset_name, image_id):
        if dataset_name == "inaturalist":
            image_path = os.path.join(self.inat21_dir, self.inat21_map[image_id])
        elif dataset_name == "landmarks":
            image_path = os.path.join(
                self.gldv2_dir, image_id[0], image_id[1], image_id[2], image_id + ".jpg"
            )
        else:
            raise ValueError("Invalid dataset name")

        return image_path

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Get the query image and question
        question = ann["question"]
        query_image_dataset = ann["image_dataset"]
        query_image_ids = ann["image_ids"]
        query_image_path = self._get_image_path(query_image_dataset, query_image_ids)
        
        query_image = Image.open(query_image_path).convert("RGB")
        query_image = self.vis_processor(query_image)
        question = self.text_processor(question)

        # Get the evidence image and evidence section
        evidence_section = ann["evidence_section"]
        evidence_image = ann["evidence_image"]

        random_index = random.randint(0, len(evidence_section) - 1)
        evidence_section = evidence_section[random_index]
        evidence_image = evidence_image[random_index]

        if len(evidence_image) > 0:
            evidence_image_path = os.path.join(self.kb_image_dir, evidence_image[0])
            evidence_image = Image.open(evidence_image_path).convert("RGB")
        else:
            # create a blank image
            evidence_image = Image.new("RGB", (224, 224), (0, 0, 0))

        evidence_image = self.vis_processor(evidence_image)
        evidence_section = self.text_processor(evidence_section)

        return {
            "question": question,
            "query_image": query_image,
            "evidence": evidence_section,
            "evidence_image": evidence_image,
        }
