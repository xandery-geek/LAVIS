
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.blip_processors import Blip2ImageTrainProcessor

__all__ = [
    "BaseProcessor",
    "Blip2ImageTrainProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
