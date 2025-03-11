from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrain, DataGenerator

#Dataloader
train_dataset = L(DataGenerator)(
    data = L(ImageFileTrain)(
        alpha_dir='/opt/data/private/lyh/Datasets/AdobeImageMatting/Train/alpha',
        fg_dir='/opt/data/private/lyh/Datasets/AdobeImageMatting/Train/fg',
        bg_dir='/opt/data/private/lyh/Datasets/coco2014/raw/train2014',
        root='/opt/data/private/lyh/Datasets/AdobeImageMatting'
    ),
    phase = 'train'
)
# 
dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset = train_dataset,
    batch_size=15,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset = train_dataset,
    ),
    drop_last=True
)