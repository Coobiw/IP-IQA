"""
 refet to https://github.com/salesforce/LAVIS
"""

from ipiqa.common.registry import registry
from ipiqa.tasks.base_task import BaseTask
from ipiqa.tasks.image2prompt import Image2PromptTask
from ipiqa.tasks.agiqa import AGIQATask
from ipiqa.tasks.agiqa_doublescore import AGIQADoubleScoresTask


def setup_task(cfg):
    assert "task" in cfg.run, "Task name must be provided."

    task_name = cfg.run.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "Image2PromptTask",
    "AGIQATask",
    "AGIQADoubleScoresTask",
]
