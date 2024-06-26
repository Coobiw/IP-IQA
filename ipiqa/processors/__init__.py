"""
 refet to https://github.com/salesforce/LAVIS
"""


from ipiqa.processors.base_processor import BaseProcessor
from ipiqa.processors.image_processor import ImageEvalProcessor, ImageTrainProcessor
from ipiqa.common.registry import registry

__all__ = [
    "BaseProcessor",
    "ImageEvalProcessor",
    "ImageTrainProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
