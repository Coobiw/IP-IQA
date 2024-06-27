from ipiqa.tasks.base_task import BaseTask
from ipiqa.common.registry import registry

import torch
import torch.nn as nn
import torch.distributed as dist

from ipiqa.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from ipiqa.common.logger import MetricLogger, SmoothedValue
from ipiqa.common.registry import registry
from ipiqa.datasets.data_utils import prepare_sample

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

import logging

@registry.register_task("agiqa")
class AGIQATask(BaseTask):
    def __init__(self,train_fn,val_fn,**kwargs):
        super().__init__(train_fn=train_fn)

        self.val_fn = val_fn

    @classmethod
    def setup_task(cls, **kwargs):
        def iqa_loss(model,samples):
            x,y,text = samples['images'], samples['mos'], samples['text']
            output = model(x,text).squeeze(dim=-1)
            criterion = nn.MSELoss()
            loss = criterion(output, y)
            loss_dict = {"loss": loss.detach().clone()}
            return loss,loss_dict

        def iqa_loss_eval(model,samples):
            x,y,text = samples['images'], samples['mos'], samples['text']
            output = model(x,text).squeeze(dim=-1)
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(output, y)
            loss_np = loss.detach().cpu().numpy().tolist()
            pred_np = output.detach().cpu().numpy().tolist()
            label_np = y.detach().cpu().numpy().tolist()
            ret = zip(loss_np,pred_np,label_np)
            return ret

        return cls(train_fn=iqa_loss,val_fn=iqa_loss_eval)

    def evaluation(self, model, data_loader, cuda_enabled=True):
        results = []

        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def after_evaluation(self, val_result, **kwargs):
        epoch = kwargs.get('epoch',None)
        pred = np.array([], dtype=np.float64)
        mos = np.array([], dtype=np.float64)
        losses = np.array([], dtype=np.float64)
        # import pdb;pdb.set_trace()
        for info in val_result:
            losses = np.append(losses, info[0])
            pred = np.append(pred, info[1])
            mos = np.append(mos, info[2])


        plcc, srocc, krocc, rmse = pearsonr(pred, mos)[0], spearmanr(pred, mos)[0], kendalltau(pred, mos)[0], np.sqrt(np.mean((pred-mos)**2))

        Loss = np.mean(losses)

        if epoch is not None:
            logging.info("EPOCH[{}] -> PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, RMSE: {:.6f}, LOSS: {:.6f}".format(epoch, plcc, srocc, krocc, rmse, Loss))
        else:
            logging.info("PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, RMSE: {:.6f}, LOSS: {:.6f}".format(plcc, srocc, krocc, rmse, Loss))

        score = srocc + plcc + krocc

        metrics = {}
        metrics["agg_metrics"] = score
        metrics['PLCC'] = plcc
        metrics['SROCC'] = srocc
        metrics['KROCC'] = krocc
        metrics['RMSE'] = rmse

        return metrics

    def valid_step(self, model, samples):
        return self.val_fn(model,samples)