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

@registry.register_task("agiqa_doublescore")
class AGIQADoubleScoresTask(BaseTask):
    def __init__(self,train_fn,val_fn,**kwargs):
        super().__init__(train_fn=train_fn)

        self.val_fn = val_fn

    @classmethod
    def setup_task(cls, **kwargs):
        def iqa_loss(model,samples):
            x,y,text = samples['images'], samples['score'], samples['text']
            output = model(x,text)
            criterion = nn.MSELoss()
            loss = criterion(output, y)
            loss_dict = {"loss": loss.detach().clone()}
            return loss,loss_dict

        def iqa_loss_eval(model,samples):
            x,y,text = samples['images'], samples['score'], samples['text']
            output = model(x,text)
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(output, y)
            loss_qual = loss.detach().cpu()[:,0].numpy().tolist()
            loss_align = loss.detach().cpu()[:,1].numpy().tolist()
            pred_qual = output.detach().cpu()[:,0].numpy().tolist()
            pred_align = output.detach().cpu()[:,1].numpy().tolist()
            label_qual = y.detach().cpu()[:,0].numpy().tolist()
            label_align = y.detach().cpu()[:,1].numpy().tolist()
            ret = zip(loss_qual,pred_qual,label_qual,loss_align,pred_align,label_align)
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
        pred_qual = np.array([], dtype=np.float64)
        mos_qual = np.array([], dtype=np.float64)
        losses_qual = np.array([], dtype=np.float64)
        pred_align = np.array([], dtype=np.float64)
        mos_align = np.array([], dtype=np.float64)
        losses_align = np.array([], dtype=np.float64)
        # import pdb;pdb.set_trace()
        for info in val_result:
            losses_qual = np.append(losses_qual, info[0])
            pred_qual = np.append(pred_qual, info[1])
            mos_qual = np.append(mos_qual, info[2])
            losses_align = np.append(losses_align, info[3])
            pred_align = np.append(pred_align, info[4])
            mos_align = np.append(mos_align, info[5])


        plcc_qual, srocc_qual, krocc_qual = pearsonr(pred_qual, mos_qual)[0], spearmanr(pred_qual, mos_qual)[0], kendalltau(pred_qual, mos_qual)[0]
        plcc_align, srocc_align, krocc_align = pearsonr(pred_align, mos_align)[0], spearmanr(pred_align, mos_align)[0], kendalltau(pred_align, mos_align)[0]

        Loss_qual = np.mean(losses_qual)
        Loss_align = np.mean(losses_align)

        if epoch is not None:
            logging.info("Qual: EPOCH[{}] -> PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, LOSS: {:.6f}".format(epoch, plcc_qual, srocc_qual, krocc_qual, Loss_qual))
            logging.info("Align: EPOCH[{}] -> PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, LOSS: {:.6f}".format(epoch, plcc_align, srocc_align, krocc_align, Loss_align))
        else:
            logging.info("Qual: PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, LOSS: {:.6f}".format(plcc_qual, srocc_qual, krocc_qual, Loss_qual))
            logging.info("Align: PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, LOSS: {:.6f}".format(plcc_align, srocc_align, krocc_align, Loss_align))

        score_qual = srocc_qual + plcc_qual + krocc_qual
        score_align = srocc_align + plcc_align + krocc_align

        metrics = {}
        metrics["agg_metrics"] = score_qual + score_align
        metrics['qual_agg'] = score_qual
        metrics['qual_PLCC'] = plcc_qual
        metrics['qual_SROCC'] = srocc_qual
        metrics['qual_KROCC'] = krocc_qual
        metrics['align_agg'] = score_align
        metrics['align_PLCC'] = plcc_align
        metrics['align_SROCC'] = srocc_align
        metrics['align_KROCC'] = krocc_align

        return metrics

    def valid_step(self, model, samples):
        return self.val_fn(model,samples)