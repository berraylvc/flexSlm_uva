# deit3_recipe.py
import math
import time
import copy
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ----------------------
# Hyperparameters (tweak to your budget)
# ----------------------
IMG_SIZE = 224
BATCH_SIZE = 1024        # scale to available GPUs / memory
NUM_WORKERS = 8
EPOCHS = 300
WARMUP_EPOCHS = 5
BASE_LR = 5e-4           # DeiT-III style
WEIGHT_DECAY = 0.05
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0       # used in same mixup function
MIXUP_PROB = 1.0
LABEL_SMOOTH = 0.1
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.99996
NUM_CLASSES = 1000       # adjust to your dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True               # use mixed precision
PRINT_FREQ = 50
SAVE_PATH = "checkpoint_deit3.pth"

# ----------------------
# Utilities: Mixup (simple) & soft CE
# ----------------------
def rand_beta(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(torch._sample_dirichlet(torch.tensor([alpha, alpha])).tolist()[0])

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns mixed inputs and soft labels (one-hot mixed).
    If alpha == 0, returns original.
    """
    if alpha <= 0:
        # return hard one-hot
        return x, F.one_hot(y, NUM_CLASSES).float().to(x.device)
    batch_size = x.size(0)
    lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()  # uses PyTorch internal Dirichlet sampling for Beta
    index = torch.randperm(batch_size, device=x.device)
    x_mixed = lam * x + (1 - lam) * x[index]
    y_a = F.one_hot(y, NUM_CLASSES).float().to(x.device)
    y_b = F.one_hot(y[index], NUM_CLASSES).float().to(x.device)
    y_mixed = lam * y_a + (1 - lam) * y_b
    return x_mixed, y_mixed

def soft_cross_entropy(pred: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    pred: logits (N, C)
    soft_targets: probabilities (N, C)
    """
    log_prob = F.log_softmax(pred, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()

def smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    assert 0.0 <= smoothing < 1.0
    with torch.no_grad():
        one_hot = F.one_hot(targets, n_classes).float().to(targets.device)
        one_hot = one_hot * (1.0 - smoothing) + smoothing / n_classes
    return one_hot

# ----------------------
# EMA helper
# ----------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.ema = copy.deepcopy(model).eval().to(next(model.parameters()).device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))

# ----------------------
# LR scheduler: warmup + cosine
# ----------------------
def create_scheduler(optimizer, epochs, warmup_epochs, base_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        # cosine decay after warmup
        t = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda)

# ----------------------
# Data transforms (train / eval)
# ----------------------
def build_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),   # (num_ops, magnitude) per DeiT-III style
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

# ----------------------
# Training and evaluation loops
# ----------------------
def train_one_epoch(model, loader, optimizer, scaler, epoch, ema: ModelEMA=None):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        # Mixup
        if MIXUP_ALPHA > 0 and torch.rand(1).item() < MIXUP_PROB:
            images, soft_targets = mixup_data(images, targets, MIXUP_ALPHA)
        else:
            # soft targets from label smoothing
            soft_targets = smooth_one_hot(targets, NUM_CLASSES, LABEL_SMOOTH)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=AMP):
            cls_logits, dist_logits = model(images)  # model returns two heads in training
            # compute soft CE for both heads
            loss_cls = soft_cross_entropy(cls_logits, soft_targets)
            loss_dist = soft_cross_entropy(dist_logits, soft_targets)
            loss = 0.5 * loss_cls + 0.5 * loss_dist

        # backward
        if AMP:
            scaler.scale(loss).backward()
            # gradient clipping
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item()
        if i % PRINT_FREQ == 0 and i > 0:
            avg = running_loss / (i + 1)
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}] Iter [{i}/{len(loader)}] loss={avg:.4f} time={elapsed:.1f}s")

    epoch_loss = running_loss / len(loader)
    return epoch_loss

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    top1 = 0
    total = 0
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=AMP):
            out = model(images)  # model.eval() -> returns averaged logits
            if isinstance(out, tuple):
                # in case model still returns tuple, average them
                out = (out[0] + out[1]) / 2
            preds = out.argmax(dim=1)
        top1 += (preds == targets).sum().item()
        total += targets.size(0)
    acc = top1 / total
    return acc

# ----------------------
# Main training entry
# ----------------------
def main(train_dir, val_dir, model):
    train_tf, val_tf = build_transforms(IMG_SIZE)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = create_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS, BASE_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    ema = ModelEMA(model, EMA_DECAY)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        # train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, ema=ema)
        scheduler.step()
        # evaluate
        # use EMA model for validation (recommended)
        val_model = ema.ema if ema is not None else model
        val_acc = evaluate(val_model, val_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.ema.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(ckpt, SAVE_PATH)
            print(f"Saved best checkpoint (acc={best_acc:.4f}) -> {SAVE_PATH}")

    print("Training finished. Best val acc:", best_acc)

# ----------------------
# Example usage:
# ----------------------
if __name__ == "__main__":
    # user should replace the following with their model and dataset paths:
    # from yourmodule import VisionTransformer  (that returns cls, dist in train)
    # model = VisionTransformer(your_config)
    # main("/path/to/imagenet/train", "/path/to/imagenet/val", model)
    pass
