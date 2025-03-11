from detectron2.engine import AMPTrainer
import torch
import time
import detectron2.utils.comm as comm
import logging

from detectron2.utils.events import EventWriter, get_event_storage

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class MattingTrainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, grad_scaler=None, log_image_period = 2000):
        super().__init__(model, data_loader, optimizer, grad_scaler=None)
        self.data_loader_iter = iter(cycle(self.data_loader))
        self.log_image_period = log_image_period


    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        #matting pass
        start = time.perf_counter()        
        data = next(self.data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict, output_images, out_pred_prob = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        if self.iter % 20 == 0:
            self.write_ratios(out_pred_prob)
        self._write_metrics(loss_dict, data_time)
        self._write_images(output_images, data)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def write_ratios(self, out_pred_prob):
        storage = get_event_storage()
        for i in range(len(out_pred_prob)):
            storage.put_scalar(f"{i}_block_token_ratio", out_pred_prob[i].sum() / out_pred_prob[i].numel(), cur_iter = self.iter)
        storage.put_scalar("total_token_ratio", sum([p.sum() / p.numel() for p in out_pred_prob]) / len(out_pred_prob), cur_iter = self.iter)
    
    def _write_images(self, output_images: torch.Tensor, data: torch.Tensor, iter: int = None):
        logger = logging.getLogger(__name__)
        iter = self.iter if iter is None else iter
        if (iter + 1) % self.log_image_period == 0:
            try:
                MattingTrainer.write_images(output_images, data, iter)
            except Exception:
                logger.exception("Exception in writing images: ")
                raise
    
    @staticmethod
    def write_images(output_images: torch.Tensor, data: torch.Tensor, cur_iter:int = None):
        # output_images = output_images.detach().cpu()
        if comm.is_main_process():
            storage = get_event_storage()
            storage.put_image("fg", data["fg"])
            storage.put_image("alpha_gt", data["alpha"])
            storage.put_image("bg", data["bg"])
            storage.put_image("trimap", data["trimap"])
            storage.put_image("image", data["image"])
            # storage._block_ratio = (block_ratio, storage.iter)
            for key in output_images.keys():
                storage.put_image(key, output_images[key])