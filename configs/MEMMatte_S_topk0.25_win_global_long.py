from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader


train.max_iter = int(43100 / 16 / 2 * 30)
train.checkpointer.period = int(43100 / 16 / 2 * 2)

model.backbone.use_rel_pos = True
model.backbone.topk = 0.25
model.backbone.window_block_indexes=[0,1,3,4,6,7,9,10,] # 2, 5, 8 11 for global attention
model.backbone.multi_score = True
model.distill = True

model.teacher_backbone.window_block_indexes=[0,1,3,4,6,7,9,10,] # 2, 5, 8 11 for global attention

optimizer.lr=5e-4
lr_multiplier.scheduler.values=[1.0, 0.1, 0.05]
lr_multiplier.scheduler.milestones=[int(43100 / 16 / 2 * 6), int(43100 / 16 / 2 * 26)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = './pretrained/ViTMatte_S_Com_with_teacher.pth'
train.output_dir = './output_of_train/ViTMatte_S_topk0.25_win_global_long'

dataloader.train.batch_size=16
dataloader.train.num_workers=2
