import torch
import math
import sys
from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

import random
import torch.nn.functional as F
import torch.nn as nn


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    kd_alpha=0.5,
    distill_type="soft",   # "soft" or "hard"
    transfer_type="progressive",  # "progressive" or "US"
):
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    
    max_level = 4

    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        optimizer.zero_grad()

        with amp_autocast():

            # =========================
            # FULL LEVEL
            # =========================
            out_full = model(im, level=max_level)
            loss_full = ce_loss_fn(out_full, seg_gt)
            
            if loss_scaler is not None:
                loss_scaler.scale(loss_full).backward()
            else:
                loss_full.backward()

            # =========================
            # 3Q LEVEL
            # =========================
            level_3q = random.randint(1, max_level // 2)
            out_3q = model(im, level=level_3q)

            if distill_type == "hard":
                kd_3q = F.cross_entropy(out_3q, out_full.detach().argmax(dim=1))
            else:
                kd_3q = kl_loss(
                    F.log_softmax(out_3q, dim=1),
                    F.softmax(out_full.detach(), dim=1),
                )

            # loss_3q = kd_alpha * kd_3q + (1 - kd_alpha) * ce_loss_fn(out_3q, seg_gt)
            loss_3q = ce_loss_fn(out_3q, seg_gt)
            
            if loss_scaler is not None:
                loss_scaler.scale(loss_3q).backward()
            else:
                loss_3q.backward()

            # =========================
            # 2Q LEVEL
            # =========================
            if transfer_type == "US":
                teacher_2q = out_full
            else:
                teacher_2q = out_3q

            level_2q = random.randint(
                max_level // 2 + 1,
                max_level - 1
            )
            out_2q = model(im, level=level_2q)

            if distill_type == "hard":
                kd_2q = F.cross_entropy(out_2q, teacher_2q.detach().argmax(dim=1))
            else:
                kd_2q = kl_loss(
                    F.log_softmax(out_2q, dim=1),
                    F.softmax(teacher_2q.detach(), dim=1),
                )

            # loss_2q = kd_alpha * kd_2q + (1 - kd_alpha) * ce_loss_fn(out_2q, seg_gt)
            loss_2q = ce_loss_fn(out_2q, seg_gt)
            
            if loss_scaler is not None:
                loss_scaler.scale(loss_2q).backward()
            else:
                loss_2q.backward()

            # =========================
            # 1Q LEVEL
            # =========================
            if transfer_type == "US":
                teacher_1q = out_full
            else:
                teacher_1q = out_2q

            out_1q = model(im, level=0)

            if distill_type == "hard":
                kd_1q = F.cross_entropy(out_1q, teacher_1q.detach().argmax(dim=1))
            else:
                kd_1q = kl_loss(
                    F.log_softmax(out_1q, dim=1),
                    F.softmax(teacher_1q.detach(), dim=1),
                )

            # loss_1q = kd_alpha * kd_1q + (1 - kd_alpha) * ce_loss_fn(out_1q, seg_gt)
            loss_1q = ce_loss_fn(out_1q, seg_gt)
            
            if loss_scaler is not None:
                loss_scaler.scale(loss_1q).backward()
            else:
                loss_1q.backward()

            # =========================
            # TOTAL LOSS
            # =========================
            loss = loss_full + loss_3q + loss_2q + loss_1q
        
            
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"Loss is {loss_val}, stopping training")
            sys.exit(1)

        # =========================
        # BACKWARD
        # =========================
        if loss_scaler is not None:
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            optimizer.step()
            

        # =========================
        # LR STEP
        # =========================
        # num_updates += 1
        # lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        # =========================
        # LOGGING
        # =========================
        logger.update(
            loss=loss.item(),
            loss_full=loss_full.item(),
            loss_3q=loss_3q.item(),
            loss_2q=loss_2q.item(),
            loss_1q=loss_1q.item(),
            lr_backbone=float(optimizer.param_groups[0]["lr"]),
            lr_decoder=float(optimizer.param_groups[1]["lr"]),
        )

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    model.eval()

    max_level = 4

    # Store predictions per level
    val_seg_pred_levels = {
        level: {} for level in range(max_level + 1)
    }

    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]

        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        for level in range(max_level + 1):

            with amp_autocast():
                seg_pred = utils.inference(
                    model_without_ddp,
                    level,
                    ims,
                    ims_metas,
                    ori_shape,
                    window_size,
                    window_stride,
                    batch_size=1,
                )
                seg_pred = seg_pred.argmax(0)

            seg_pred = seg_pred.cpu().numpy()
            val_seg_pred_levels[level][filename] = seg_pred

    # =========================
    # Compute metrics per level
    # =========================
    results = {}

    for level in range(max_level + 1):
        preds = gather_data(val_seg_pred_levels[level])

        scores = compute_metrics(
            preds,
            val_seg_gt,
            data_loader.unwrapped.n_cls,
            ignore_index=IGNORE_LABEL,
            distributed=ptu.distributed,
        )

        # Log per-level metrics
        for k, v in scores.items():
            logger.update(**{f"{k}_level{level}": v, "n": 1})

        results[level] = scores

    return logger, results