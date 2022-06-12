import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorLoss(nn.Module):
    def __init__(self, input_shape=[416, 416], reduction='none'):
        super(VectorLoss, self).__init__()
        self.reduction = reduction
        self.input_shape = input_shape

    def forward(self, preds, target):
        assert preds.shape[0] == target.shape[0]
        loss = torch.zeros(preds.shape)
        loss[..., 0] = torch.abs_((target[..., 0] - preds[..., 0]) / self.input_shape[0])
        loss[..., 1] = torch.abs_((target[..., 1] - preds[..., 1]) / self.input_shape[1])
        loss[..., 2:] = torch.abs_((target[..., 2:] - preds[..., 2:]))*10
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class DPSV_Loss(nn.Module):
    def __init__(self, num_classes, input_shape, strides=[8, 16, 32]):
        super(DPSV_Loss, self).__init__()
        self.num_classes = num_classes
        self.srides = strides

        self.bce_loss = nn.BCELoss(reduction="none")
        self.vector_loss = VectorLoss(input_shape=input_shape, reduction="none")
        self.grids = [torch.zeros(1)] * len(strides)
        self.input_shape = input_shape

    def forward(self, inputs, labels=None):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        # -----------------------------------------------#
        # inputs    [[batch_size, num_classes + 4, 20, 20]
        #            [batch_size, num_classes + 4, 40, 40]
        #            [batch_size, num_classes + 4, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 4]
        #            [batch_size, 1600, num_classes + 4]
        #            [batch_size, 6400, num_classes + 4]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        # -----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.srides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)
        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize))
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        output[..., :2] = (output[..., :2] + grid) * stride
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        # 从output中提取出vector，cls，obj的预测结果
        # [batchsieze,n_grids,3]
        vec_preds = outputs[:, :, :4]
        # [batchsieze,n_grids,1]
        obj_preds = outputs[:, :, 4:5]
        # [batchsieze, n_grids, num_classes]
        cls_preds = outputs[:, :, 5:]
        total_num_vecs = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        vec_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                vec_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_vecs, 1))
                fg_mask = outputs.new_zeros(total_num_vecs).bool()
            else:
                gt_vec_per_image = labels[batch_idx][..., :4]
                gt_cls_per_image = labels[batch_idx][..., 4]
                pred_vec_per_image = vec_preds[batch_idx]
                pred_cls_per_image = cls_preds[batch_idx]
                fg_mask, gt_matched_vecs, gt_matched_class, pred_matched_vec, pred_matched_class, num_fg_img = self.get_assignments(
                    num_gt, gt_vec_per_image, gt_cls_per_image, pred_vec_per_image, pred_cls_per_image,
                    x_shifts, y_shifts, expanded_strides)
                torch.cuda.empty_cache()
                num_fg = num_fg + num_fg_img
                cls_target = F.one_hot(gt_matched_class.to(torch.int64),
                                       self.num_classes).float()
                obj_target = fg_mask.unsqueeze(-1)
                vec_target = gt_matched_vecs
            cls_targets.append(cls_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            vec_targets.append(vec_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        vec_targets = torch.cat(vec_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        a = self.vector_loss(vec_preds.view(-1, 4)[fg_masks], vec_targets)
        loss_vec = (self.vector_loss(vec_preds.view(-1, 4)[fg_masks], vec_targets)).sum() * 10000
        obj_targets = obj_targets.float()
        loss_obj = -(1000 * obj_targets * torch.log(obj_preds.view(-1, 1) + 1e-8) + (1 - obj_targets) * torch.log(
            1 - obj_preds.view(-1, 1) + 1e-8)).sum()

        loss_cls = (self.bce_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() * 500
        # loss = 500 * loss_vec + loss_obj + loss_cls
        # if num_fg == 0:
        #     print("num_fg=0 error")
        # print("obj_loss:%s\tvec_Loss:%s\tcls_loss:%s" % (loss_obj_1, loss_vec, loss_cls))
        if torch.isnan(loss_obj) != 0:
            print("error")
            a = torch.isnan(obj_preds)
        return loss_obj + loss_vec + loss_cls, a.sum(dim=0)

    def get_assignments(self, num_gt, gt_vec_per_image, gt_cls_per_image, pred_vec_per_image, pred_cls_per_image,
                        x_shifts, y_shifts, expanded_strides):
        fg_mask, is_in_range = self.get_candidate_info(gt_vec_per_image, expanded_strides=expanded_strides,
                                                       x_shifts=x_shifts, y_shifts=y_shifts, num_gt=num_gt)
        matched_gt_inds = is_in_range.float().argmax(0)
        gt_matched_class = gt_cls_per_image[matched_gt_inds]
        gt_matched_vecs = gt_vec_per_image[matched_gt_inds]
        pred_matched_class = pred_cls_per_image[fg_mask]
        pred_matched_vec = pred_vec_per_image[fg_mask]
        num_fg = pred_matched_vec.shape[0]
        return fg_mask, gt_matched_vecs, gt_matched_class, pred_matched_vec, pred_matched_class, num_fg

    def get_candidate_info(self, gt_vectors_per_image, expanded_strides, x_shifts, y_shifts, num_gt,
                           center_r=0.5):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        gt_vectors_per_image_l = gt_vectors_per_image[:, 0].view(-1, 1) - center_r * expanded_strides_per_image
        gt_vectors_per_image_r = gt_vectors_per_image[:, 0].view(-1, 1) + center_r * expanded_strides_per_image
        gt_vectors_per_image_t = gt_vectors_per_image[:, 1].view(-1, 1) - center_r * expanded_strides_per_image
        gt_vectors_per_image_b = gt_vectors_per_image[:, 1].view(-1, 1) + center_r * expanded_strides_per_image

        b_l = x_centers_per_image - gt_vectors_per_image_l
        b_r = gt_vectors_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_vectors_per_image_t
        b_b = gt_vectors_per_image_b - y_centers_per_image
        vector_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_range = vector_deltas.min(dim=-1).values > 0.0
        is_in_range_all = is_in_range.sum(dim=0) > 0

        return is_in_range_all, is_in_range[:, is_in_range_all]


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    test = VectorLoss()
    test_pred = torch.Tensor([[[15, 18, 0.2], [18, 15, 3]], [[15, 18, 0.2], [18, 15, 3]]])
    test_target = torch.Tensor([[[15, 15, 0.2], [18, 18, 3]], [[15, 15, 0.2], [18, 18, 3]]])
    test_loss = test.forward(preds=test_pred, target=test_target)
    print(test_loss)
