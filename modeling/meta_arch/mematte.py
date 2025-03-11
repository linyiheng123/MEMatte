import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from detectron2.structures import ImageList

class MEMatte(nn.Module):
    def __init__(self,
                 *,
                 teacher_backbone,
                 backbone,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 distill = True,
                 distill_loss_ratio = 1.,
                 token_loss_ratio = 1.,
                 balance_loss = "MSE",
                 ):
        super(MEMatte, self).__init__()
        self.teacher_backbone = teacher_backbone
        self.backbone = backbone
        self.criterion = criterion
        self.input_format = input_format
        self.balance_loss = balance_loss
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.distill = distill
        self.distill_loss_ratio = distill_loss_ratio
        self.token_loss_ratio = token_loss_ratio
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, patch_decoder=True):
        images, targets, H, W = self.preprocess_inputs(batched_inputs)

        if self.training:
            if self.distill == True:
                self.teacher_backbone.eval()
                features, out_pred_prob = self.backbone(images)
                teacher_features = self.teacher_backbone(images)
                distill_loss = F.mse_loss(features, teacher_features)
            else:
                features, out_pred_prob = self.backbone(images)
            outputs = self.decoder(features, images)
            assert targets is not None
            trimap = images[:, 3:4]
            sample_map = torch.zeros_like(trimap)
            sample_map[trimap==0.5] = 1 # 2*B*dim*(2*ratio*hw*dim + ratio*hw*ratio*hw)
            losses = self.criterion(sample_map ,outputs, targets)
            if self.distill:
                losses['distill_loss'] = distill_loss * self.distill_loss_ratio
            total_ratio = sum([p.sum() / p.numel() for p in out_pred_prob]) / len(out_pred_prob)
            losses['mse_ratio_loss'] = self.token_loss_ratio * F.mse_loss(total_ratio, torch.tensor(self.backbone.topk).cuda())

            tensorboard_images = dict()
            tensorboard_images['pred_alpha'] = outputs['phas']
            return losses, tensorboard_images, out_pred_prob
        else:
            features, out_pred_prob, out_hard_keep_decision = self.backbone(images)
            if patch_decoder:
                outputs = self.patch_inference(features=features, images=images)
            else:
                outputs = self.decoder(features, images)
    
            outputs['phas'] = outputs['phas'][:,:,:H,:W]

            return outputs, out_pred_prob, out_hard_keep_decision
            
            """测试flops"""
            # features = self.backbone(images)
            # # outputs = self.decoder(features, images)
            # # outputs['phas'] = outputs['phas'][:,:,:H,:W]
            # return features

            """测试decoder flops"""

        


    def patch_inference(self, features, images):
        patch_size = 512
        overlap = 64
        image_size = patch_size + 2 * overlap
        feature_patch_size = patch_size // 16
        feature_overlap = overlap // 16
        features_size = feature_patch_size + 2 * feature_overlap
        B, C, H, W = images.shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        pad_images = F.pad(images.permute(0,2,3,1), (0,0,0,pad_w,0,pad_h)).permute(0,3,1,2)
        _, _, pad_H, pad_W = pad_images.shape

        _, _, H_fea, W_fea = features.shape
        pad_fea_h = (feature_patch_size - H_fea % feature_patch_size) % feature_patch_size
        pad_fea_w = (feature_patch_size - W_fea % feature_patch_size) % feature_patch_size
        pad_features = F.pad(features.permute(0,2,3,1), (0,0,0,pad_fea_w,0,pad_fea_h)).permute(0,3,1,2)
        _, _, pad_fea_H, pad_fea_W = pad_features.shape
        
        h_patch_num = pad_images.shape[2] // patch_size
        w_patch_num = pad_images.shape[3] // patch_size

        outputs = torch.zeros_like(pad_images[:,0:1,:,:])

        for i in range(h_patch_num):
            for j in range(w_patch_num):
                start_top = i * patch_size
                end_bottom = start_top + patch_size
                start_left = j*patch_size
                end_right = start_left + patch_size
                coor_top = start_top if (start_top - overlap) < 0 else (start_top - overlap)
                coor_bottom = end_bottom if (end_bottom + overlap) > pad_H else (end_bottom + overlap)
                coor_left = start_left if (start_left - overlap) < 0 else (start_left - overlap)
                coor_right = end_right if (end_right + overlap) > pad_W else (end_right + overlap)
                selected_images = pad_images[:,:,coor_top:coor_bottom, coor_left:coor_right]

                fea_start_top = i * feature_patch_size
                fea_end_bottom = fea_start_top + feature_patch_size
                fea_start_left = j*feature_patch_size
                fea_end_right = fea_start_left + feature_patch_size
                coor_top_fea = fea_start_top if (fea_start_top - feature_overlap) < 0 else (fea_start_top - feature_overlap)
                coor_bottom_fea = fea_end_bottom if (fea_end_bottom + feature_overlap) > pad_fea_H else (fea_end_bottom + feature_overlap)
                coor_left_fea = fea_start_left if (fea_start_left - feature_overlap) < 0 else (fea_start_left - feature_overlap)
                coor_right_fea = fea_end_right if (fea_end_right + feature_overlap) > pad_fea_W else (fea_end_right + feature_overlap)
                selected_fea = pad_features[:,:,coor_top_fea:coor_bottom_fea, coor_left_fea:coor_right_fea]



                outputs_patch = self.decoder(selected_fea, selected_images)

                coor_top = start_top if (start_top - overlap) < 0 else (coor_top + overlap)
                coor_bottom = coor_top + patch_size
                coor_left = start_left if (start_left - overlap) < 0 else (coor_left + overlap)
                coor_right = coor_left + patch_size

                coor_out_top = 0 if (start_top - overlap) < 0 else overlap
                coor_out_bottom = coor_out_top + patch_size
                coor_out_left = 0 if (start_left - overlap) < 0 else overlap
                coor_out_right = coor_out_left + patch_size

                outputs[:, :, coor_top:coor_bottom, coor_left:coor_right] = outputs_patch['phas'][:,:,coor_out_top:coor_out_bottom,coor_out_left:coor_out_right]

        outputs = outputs[:,:,:H, :W]
        return {'phas':outputs}

    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        trimap = batched_inputs['trimap'].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std

        if 'fg' in batched_inputs.keys():
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

        images = torch.cat((images, trimap), dim=1)
        
        B, C, H, W = images.shape
        if images.shape[-1]%32!=0 or images.shape[-2]%32!=0:
            new_H = (32-images.shape[-2]%32) + H
            new_W = (32-images.shape[-1]%32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]
            del images
            images = new_images

        if "alpha" in batched_inputs:
            phas = batched_inputs["alpha"].to(self.device)
        else:
            phas = None

        return images, dict(phas=phas), H, W

