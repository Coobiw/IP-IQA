from ipiqa.tasks.base_task import BaseTask
from ipiqa.common.registry import registry

import torch
import torch.nn as nn

@registry.register_task("image2prompt")
class Image2PromptTask(BaseTask):
    def __init__(self,train_fn):
        super().__init__(train_fn=train_fn)

    @classmethod
    def setup_task(cls, **kwargs):
        def image2prompt_loss(model,samples):
            x,y = samples['images'], samples['prompt_embeddings']
            output = model(x)
            criterion = nn.CosineEmbeddingLoss()
            target = torch.ones(x.size(0)).to(x.device)
            loss = criterion(output, y, target)
            loss_dict = {"loss": loss.detach().clone()}
            return loss,loss_dict
        return cls(train_fn=image2prompt_loss)

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass