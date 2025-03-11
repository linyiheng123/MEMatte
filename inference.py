import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser

import warnings
warnings.filterwarnings('ignore')

#Dataset and Dataloader
def collate_fn(batched_inputs):
    rets = dict()
    for k in batched_inputs[0].keys():
        rets[k] = torch.stack([_[k] for _ in batched_inputs])
    return rets

class Composition_1k(Dataset):
    def __init__(self, data_dir, finished_list = None):
        self.data_dir = data_dir
        if "AIM" in data_dir:
            self.file_names = sorted(os.listdir(opj(self.data_dir, 'original')))
        else:
            self.file_names = sorted(os.listdir(opj(self.data_dir, 'merged')), reverse=True)
       
        # self.file_names = list(set(self.file_names).difference(set(finished_list))) # difference

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if "AIM" in self.data_dir:
            tris = Image.open(opj(self.data_dir, 'trimap', self.file_names[idx].replace('jpg','png')))
            imgs = Image.open(opj(self.data_dir, 'original', self.file_names[idx]))
        else:
            tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx].replace('jpeg','png').replace('jpg','png')))
            imgs = Image.open(opj(self.data_dir, 'merged', self.file_names[idx]))
        
        sample = {}

        sample['trimap'] = F.to_tensor(tris)[0:1, :, :]
        sample['image'] = F.to_tensor(imgs)
        sample['image_name'] = self.file_names[idx]
        return sample


#model and output
def matting_inference(
    config_dir='',
    checkpoint_dir='',
    inference_dir='',
    data_dir='',
    rank=None,
    max_number_token = 18500,
):

    #initializing model
    cfg = LazyConfig.load(config_dir)
    cfg.model.teacher_backbone = None
    cfg.model.backbone.max_number_token = max_number_token
    model = instantiate(cfg.model)
    model.to(cfg.train.device if rank is None else rank)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_dir)

    #initializing dataset
    composition_1k_dataloader = DataLoader(
    dataset = Composition_1k(
        data_dir = data_dir,
    ),
    shuffle = False,
    batch_size = 1,
    )
    
    #inferencing
    os.makedirs(inference_dir, exist_ok=True)

    for data in tqdm(composition_1k_dataloader):
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)

            output, _, _ = model(data, patch_decoder=True)
            output = output['phas'].flatten(0, 2)
            trimap = data['trimap'].squeeze(0).squeeze(0)
            output[trimap == 0] = 0
            output[trimap == 1] = 1
            output = F.to_pil_image(output)
            output.save(opj(inference_dir, data['image_name'][0].replace('.jpg', '.png')))
            torch.cuda.empty_cache()

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--inference-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max-number-token', type=int, required=True, default=18500)
    
    args = parser.parse_args()
    matting_inference(
        config_dir = args.config_dir,
        checkpoint_dir = args.checkpoint_dir,
        inference_dir = args.inference_dir,
        data_dir = args.data_dir,
        max_number_token = args.max_number_token
    )
