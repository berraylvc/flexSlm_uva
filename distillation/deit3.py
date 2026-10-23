# deit3_flex.py
# Combined model + trainer implementing DeiT-3 recipe + DropPath + LayerScale
# Assumes timm, pytorch_lightning, torchvision available.

import copy
import math
import random
import dataclasses
import time
from typing import Iterable, Optional, Callable, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.data import Mixup
from timm.utils import ModelEma
from torchvision import transforms
from torchvision.transforms import RandAugment

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# ---- Replace these with your project's modules ----
import flex_modules.module as fm        # your flexible modules (ClassTokenLayer, LayerNorm, SelfAttention, Conv2d, MLPBlock, LinearSelect, etc.)
import networks.modules as vmod         # base (non-flex) modules; used by copy_to_base / load_from_base
import utils                             # your utils (get_device, get_num_nodes, save_model)
from training import TrainingContext, FlexModelConfig  # adapt paths if necessary
# ---------------------------------------------------

# ---------------------------
# Flexible DistillTokenLayer (level-aware)
# ---------------------------
class DistillTokenLayer(fm.Module):
    def __init__(self, hidden_dim: Iterable[int]):
        super().__init__()
        assert all(h > 0 for h in hidden_dim)
        assert max(hidden_dim) == hidden_dim[-1]
        self.hidden_dims = hidden_dim
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim[-1]))
        self.level = self.max_level()

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        batch_dist = self.token[:, :, :self.hidden_dims[self.level]].expand(n, -1, -1)
        # Append distillation token (after class token & patches)
        x = torch.cat([x, batch_dist], dim=1)
        return x

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Module]:
        return vmod.DistillTokenLayer

    @torch.no_grad()
    def copy_to_base(self, dest: vmod.DistillTokenLayer) -> None:
        dest.token.data = self.token.data[:, :, : self.hidden_dims[self.level]]

    @torch.no_grad()
    def load_from_base(self, src: vmod.DistillTokenLayer) -> None:
        self.token.data[:, :, : self.hidden_dims[self.level]] = src.token.data

    def _make_reg_layer(self) -> nn.Module:
        return vmod.DistillTokenLayer(self.hidden_dims[self.level])

    @torch.no_grad()
    def export_level_delta(self):
        from flex_modules.module import DownDelta, UpDelta
        return (
            DownDelta(self.hidden_dims[self.level]),
            UpDelta(self.token.data[:, :, self.hidden_dims[self.level - 1] : self.hidden_dims[self.level]])
        )

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: vmod.DistillTokenLayer, level_delta):
        model.token.data = model.token.data[:, :, : level_delta.delta].to(model.token.data)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: vmod.DistillTokenLayer, level_delta):
        model.token.data = torch.cat([model.token.data, level_delta.delta.to(model.token.data)], dim=2)

# ---------------------------
# Base DistillTokenLayer (networks.modules) -- if you need to add to networks/modules.py:
# class DistillTokenLayer(nn.Module):
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#     def forward(self, x, n):
#         dist = self.token.expand(n, -1, -1)
#         return torch.cat([x, dist], dim=1)
# ---------------------------

# ---------------------------
# EncoderBlock adapted for DeiT-3: DropPath + LayerScale
# ---------------------------
class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-4
    ):
        super().__init__()
        self.ln_1 = fm.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = fm.SelfAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.ln_2 = fm.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = fm.MLPBlock(hidden_dim, mlp_dim, dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # LayerScale parameters: one per channel
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # attention branch
        residual = x
        x = self.ln_1(x)
        x = self.self_attention(x)
        x = self.attn_dropout(x)
        # apply LayerScale: multiply per-channel then drop_path then add residual
        x = (self.layer_scale_1 * x)
        x = residual + self.drop_path(x)

        # mlp branch
        residual = x
        y = self.ln_2(x)
        y = self.mlp(y)
        y = (self.layer_scale_2 * y)
        y = residual + self.drop_path(y)

        return y

# ---------------------------
# Encoder stack
# ---------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-4
    ):
        super().__init__()
        self.pos_embedding = fm.PositionalEmbedding(seq_length, hidden_dim)
        # linear schedule for drop_path_rate across layers
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dpr = drop_path_rate * float(i) / max(1.0, num_layers - 1)
            self.blocks.append(
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr,
                    layer_scale_init_value=layer_scale_init_value
                )
            )

    def forward(self, x: torch.Tensor):
        # x shape (N, S, D)
        x = x + self.pos_embedding(x)
        for blk in self.blocks:
            x = blk(x)
        return x

# ---------------------------
# Vision Transformer (always distilled)
# ---------------------------
class VisionTransformerDistilled(fm.FlexModel):
    """
    Vision Transformer with always-on distillation token & two heads.
    Training -> returns (cls_logits, dist_logits)
    Eval -> returns averaged logits
    """

    def __init__(self, config):
        # config expected to contain: structure (image_size, patch_size, num_layers),
        # num_heads, hidden_dims (int or list), mlp_dims (int or list), num_classes, dropout, attention_dropout,
        # drop_path_rate, layer_scale_init_value, prebuilt (optional)
        super().__init__(config)

        image_size = config.structure.image_size
        patch_size = config.structure.patch_size
        num_layers = config.structure.num_layers
        num_heads = config.num_heads
        hidden_dim = config.hidden_dims  # expecting single int or list where last is active
        mlp_dim = config.mlp_dims

        num_classes = config.num_classes
        dropout = config.dropout
        attention_dropout = config.attention_dropout
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 1e-4)

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes

        # tokens
        self.class_token = fm.ClassTokenLayer(hidden_dim)
        self.dist_token = DistillTokenLayer(hidden_dim)

        # patch embedding conv projection (flexible)
        self.conv_proj = fm.Conv2d([3] * len(hidden_dim), hidden_dim, kernel_size=patch_size, stride=patch_size)

        # sequence length (patches + cls + dist)
        seq_length = (image_size // patch_size) ** 2
        seq_length += 1  # class token
        seq_length += 1  # dist token

        # encoder
        # If hidden_dim is a list of levels, use last as active hidden size for encoder creation
        active_hidden = self.hidden_dim[-1] if isinstance(self.hidden_dim, (list, tuple)) else self.hidden_dim
        active_mlp = self.mlp_dim[-1] if isinstance(self.mlp_dim, (list, tuple)) else self.mlp_dim

        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=active_hidden,
            mlp_dim=active_mlp,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value
        )

        # heads: classification and distillation head
        self.heads = nn.ModuleDict({
            "head": fm.LinearSelect(self.hidden_dim, [num_classes] * len(self.hidden_dim)),
            "head_dist": fm.LinearSelect(self.hidden_dim, [num_classes] * len(self.hidden_dim))
        })

        self.set_level_use(self.max_level())
        self.level = self.max_level()

        # handle pretrained - preserve old logic if needed
        if getattr(config, "prebuilt", None) is not None and config.prebuilt != getattr(config, "noprebuild", None):
            try:
                prebuilt = KNOWN_MODEL_PRETRAINED[(config.structure, config.prebuilt)]()
            except Exception:
                prebuilt = None
            if prebuilt is not None:
                utils.flexible_model_copy(prebuilt, self)
                self.class_token.token = copy.deepcopy(prebuilt.class_token)
                self.encoder.pos_embedding.embedding = copy.deepcopy(prebuilt.encoder.pos_embedding)

        # adjust heads for custom num_classes
        if config.num_classes != getattr(config, "DEFAULT_NUM_CLASSES", 1000):
            self.heads["head"] = fm.LinearSelect(self.hidden_dim, [num_classes] * len(self.hidden_dim))
            self.heads["head_dist"] = fm.LinearSelect(self.hidden_dim, [num_classes] * len(self.hidden_dim))

    @staticmethod
    def base_type() -> type[nn.Module]:
        return networks.vit.VisionTransformer  # adapt if network base is different

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dim) - 1

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)  # (N, D, n_h, n_w) flexible conv
        x = x.reshape(n, self.hidden_dim[self.current_level()], n_h * n_w)
        x = x.permute(0, 2, 1)  # (N, S, D)
        return x

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]

        x = self.class_token(x, n)   # prepend class token
        x = self.dist_token(x, n)    # append distillation token

        x = self.encoder(x)

        cls_out = self.heads["head"](x[:, 0])    # class token position
        dist_out = self.heads["head_dist"](x[:, -1])  # dist token pos

        if self.training:
            return cls_out, dist_out
        else:
            return (cls_out + dist_out) / 2.0

# ---------------------------
# Distillation loss (adapted from your original)
# ---------------------------
class DistillationLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, teacher_model: Optional[nn.Module],
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs: torch.Tensor, outputs, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("Model must return (cls_logits, dist_logits) for distillation.")

        with torch.no_grad():
            teacher_outputs = None
            if self.teacher_model is not None:
                teacher_outputs = self.teacher_model(inputs)
            # if no external teacher and distillation_type=='soft', fallback to self-supervision handled outside

        if self.distillation_type == 'soft':
            T = self.tau
            # KL between dist logits and teacher logits (log_target=True since teacher provided as log_softmax)
            target = F.log_softmax(teacher_outputs / T, dim=1) if teacher_outputs is not None else F.log_softmax(outputs / T, dim=1)
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                target,
                reduction='batchmean',
                log_target=True
            ) * (T * T)
        else:  # hard
            teacher_labels = teacher_outputs.argmax(dim=1) if teacher_outputs is not None else outputs.argmax(dim=1)
            distillation_loss = F.cross_entropy(outputs_kd, teacher_labels)

        loss = base_loss * (1.0 - self.alpha) + distillation_loss * self.alpha
        return loss

# ---------------------------
# DeiT-3 Lightning Trainer (adapted)
# ---------------------------
@dataclasses.dataclass
class ScalaDistillContextDeit3(TrainingContext):
    # Extend or override fields you need. Example defaults below:
    batch_size: int = 1024
    epochs: int = 300
    warmup_epochs: int = 5
    base_lr: float = 5e-4
    weight_decay: float = 0.05
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.1
    layer_scale_init_value: float = 1e-4
    ema_decay: float = 0.99996
    use_ema: bool = True
    lr_min: float = 1e-6
    randaugment_magnitude: int = 9
    randaugment_ops: int = 2
    re_prob: float = 0.25
    re_mode: str = 'pixel'
    re_count: int = 1
    precision: str = 'bf16-mixed'

class ScalaDistillTrainerDeit3(pl.LightningModule):
    def __init__(self, model_config: FlexModelConfig, training_context: ScalaDistillContextDeit3):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config', 'training_context'])
        self.model_config = model_config
        self.context = training_context

        # create submodel with drop_path param forwarded
        model_kwargs = dict(
            drop_path_rate=self.context.drop_path_rate,
            layer_scale_init_value=self.context.layer_scale_init_value
        )
        # Expect make_model to accept kwargs, otherwise adapt accordingly
        self.submodel = self.model_config.make_model(**model_kwargs).to(utils.get_device())

        # teacher (optional). For DeiT-3 teacher is typically None.
        self.teacher = None
        if getattr(self.context, "teacher_loader", None):
            try:
                self.teacher = self.context.teacher_loader()
                self.teacher.to(utils.get_device())
                self.teacher.eval()
            except Exception:
                self.teacher = None

        base_criterion = nn.CrossEntropyLoss(label_smoothing=self.context.label_smoothing)
        self.criterion = DistillationLoss(base_criterion, self.teacher, getattr(self.context, "distillation_type", "none"),
                                          getattr(self.context, "distillation_alpha", 0.5),
                                          getattr(self.context, "distillation_tau", 1.0))

        # mixup helper (timm Mixup)
        self.mixup_fn = None
        if self.context.mixup_alpha > 0 or self.context.cutmix_alpha > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=self.context.mixup_alpha,
                cutmix_alpha=self.context.cutmix_alpha,
                prob=self.context.mixup_prob,
                switch_prob=0.5,
                label_smoothing=self.context.label_smoothing,
                num_classes=self.model_config.num_classes
            )

        # EMA
        self.model_ema = ModelEma(self.submodel, decay=self.context.ema_decay) if self.context.use_ema else None

        # manual optimization for multi-step backward like your previous code
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        return self.submodel(x)

    def configure_optimizers(self):
        # Scale LR linearly with batch size (common rule)
        bs = getattr(self.context, "batch_size", 1024)
        scaled_lr = self.context.base_lr * (bs / 512.0)
        optimizer = torch.optim.AdamW(self.submodel.parameters(), lr=scaled_lr, betas=(0.9, 0.999), weight_decay=self.context.weight_decay)
        # Cosine with warmup handled manually via Lambda below
        def lr_lambda(epoch):
            if epoch < self.context.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.context.warmup_epochs))
            t = float(epoch - self.context.warmup_epochs) / float(max(1, self.context.epochs - self.context.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}

    # build transforms for dataloaders usage externally
    @staticmethod
    def build_transforms(img_size=224, context: ScalaDistillContextDeit3 = None):
        if context is None:
            context = ScalaDistillContextDeit3()
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=context.randaugment_ops, magnitude=context.randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return train_tf, val_tf

    def training_step(self, batch, batch_idx):
        # This function implements your progressive multi-level distillation
        samples, targets = batch
        device = utils.get_device()
        samples = samples.to(device)
        targets = targets.to(device)

        # Mixup support
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)

        opt = self.optimizers()
        opt.zero_grad()

        # Full model (highest level)
        self.submodel.set_level_use(self.submodel.max_level())
        outputs = self.submodel(samples)
        loss_full = self.criterion(samples, outputs, targets)
        self.manual_backward(loss_full)
        loss_full_item = loss_full.detach().cpu()

        # choose token index: 0 -> class token, 1 -> dist token (as this model appends dist token at end)
        token = 1  # dist token index offset (we'll use outputs tuple structure elsewhere)

        # compute pl-like CE target choice
        ce_targets = targets  # if needed could be teacher-provided

        # 3q model (low level random)
        level_3q = random.randint(1, max(1, self.submodel.max_level() // 2))
        self.submodel.set_level_use(level_3q)
        out_3q = self.submodel(samples)
        # out_3q may be tuple
        if getattr(self.context, "distillation_type", "none") == "none":
            # outputs are logits directly, compare full outputs
            teacher_logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            loss_3q = F.kl_div(F.log_softmax(out_3q, dim=1), F.softmax(teacher_logits.detach(), dim=1), reduction='batchmean')
            loss_3q += self.context.ce_coefficient * F.cross_entropy(out_3q, ce_targets)
        else:
            # use token-wise distillation (model returns tuple during training)
            if isinstance(out_3q, tuple):
                out_3q_cls, out_3q_dist = out_3q
            else:
                out_3q_cls, out_3q_dist = out_3q, out_3q
            if isinstance(outputs, tuple):
                out_full_cls, out_full_dist = outputs
            else:
                out_full_cls, out_full_dist = outputs, outputs
            if getattr(self.context, "distill_type", "soft") == "hard":
                loss_3q = F.cross_entropy(out_3q_dist, out_full_dist.detach().argmax(dim=1))
            else:
                loss_3q = F.kl_div(F.log_softmax(out_3q_dist, dim=1), F.softmax(out_full_dist.detach(), dim=1), reduction='batchmean')
            loss_3q += self.context.ce_coefficient * F.cross_entropy(out_3q_cls, ce_targets)
        self.manual_backward(loss_3q)
        loss_3q_item = loss_3q.detach().cpu()

        # 2q model (mid level)
        if getattr(self.context, "transfer_type", "progressive") == "US":
            teacher_2q = outputs
        else:
            teacher_2q = out_3q

        level_2q = random.randint(max(1, self.submodel.max_level() // 2 + 1), max(1, self.submodel.max_level() - 1))
        self.submodel.set_level_use(level_2q)
        out_2q = self.submodel(samples)
        if getattr(self.context, "distillation_type", "none") == "none":
            teacher_logits = teacher_2q if isinstance(teacher_2q, torch.Tensor) else (teacher_2q[0] if isinstance(teacher_2q, tuple) else teacher_2q)
            loss_2q = F.kl_div(F.log_softmax(out_2q, dim=1), F.softmax(teacher_logits.detach(), dim=1), reduction='batchmean')
            loss_2q += self.context.ce_coefficient * F.cross_entropy(out_2q, ce_targets)
        else:
            if isinstance(out_2q, tuple):
                out_2q_cls, out_2q_dist = out_2q
            else:
                out_2q_cls, out_2q_dist = out_2q, out_2q
            if isinstance(teacher_2q, tuple):
                teacher_cls, teacher_dist = teacher_2q
            else:
                teacher_cls, teacher_dist = teacher_2q, teacher_2q
            if getattr(self.context, "distill_type", "soft") == "hard":
                loss_2q = F.cross_entropy(out_2q_dist, teacher_dist.detach().argmax(dim=1))
            else:
                loss_2q = F.kl_div(F.log_softmax(out_2q_dist, dim=1), F.softmax(teacher_dist.detach(), dim=1), reduction='batchmean')
            loss_2q += self.context.ce_coefficient * F.cross_entropy(out_2q_cls, ce_targets)
        self.manual_backward(loss_2q)
        loss_2q_item = loss_2q.detach().cpu()

        # 1q model (lowest level)
        if getattr(self.context, "transfer_type", "progressive") == "US":
            teacher_1q = outputs
        else:
            teacher_1q = out_2q

        self.submodel.set_level_use(0)
        out_1q = self.submodel(samples)
        if getattr(self.context, "distillation_type", "none") == "none":
            teacher_logits = teacher_1q if isinstance(teacher_1q, torch.Tensor) else (teacher_1q[0] if isinstance(teacher_1q, tuple) else teacher_1q)
            loss_1q = F.kl_div(F.log_softmax(out_1q, dim=1), F.softmax(teacher_logits.detach(), dim=1), reduction='batchmean')
            loss_1q += self.context.ce_coefficient * F.cross_entropy(out_1q, ce_targets)
        else:
            if isinstance(out_1q, tuple):
                out_1q_cls, out_1q_dist = out_1q
            else:
                out_1q_cls, out_1q_dist = out_1q, out_1q
            if isinstance(teacher_1q, tuple):
                t_cls, t_dist = teacher_1q
            else:
                t_cls, t_dist = teacher_1q, teacher_1q
            if getattr(self.context, "distill_type", "soft") == "hard":
                loss_1q = F.cross_entropy(out_1q_dist, t_dist.detach().argmax(dim=1))
            else:
                loss_1q = F.kl_div(F.log_softmax(out_1q_dist, dim=1), F.softmax(t_dist.detach(), dim=1), reduction='batchmean')
            loss_1q += self.context.ce_coefficient * F.cross_entropy(out_1q_cls, ce_targets)
        self.manual_backward(loss_1q)
        loss_1q_item = loss_1q.detach().cpu()

        # Step optimizer once
        opt.step()

        # EMA update
        if self.model_ema is not None:
            self.model_ema.update(self.submodel)

        # Logging
        total_loss = loss_full_item + loss_3q_item + loss_2q_item + loss_1q_item
        self.log("train_loss_full", loss_full_item, prog_bar=False, sync_dist=True)
        self.log(f"train_level{level_3q}_loss", loss_3q_item, prog_bar=False, sync_dist=True)
        self.log(f"train_level{level_2q}_loss", loss_2q_item, prog_bar=False, sync_dist=True)
        self.log("train_level0_loss", loss_1q_item, prog_bar=False, sync_dist=True)
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(utils.get_device())
        y = y.to(utils.get_device())
        total_loss = 0.0
        for i in range(self.submodel.max_level() + 1):
            self.submodel.set_level_use(i)
            logits = self.submodel(x)
            # If model returns tuple in eval (averaged), ensure single tensor
            if isinstance(logits, tuple):
                logits = (logits[0] + logits[1]) / 2.0
            loss = F.cross_entropy(logits, y, label_smoothing=self.context.label_smoothing)
            acc = (logits.argmax(1) == y).float().mean()
            self.log(f"val_level{i}_loss", loss, prog_bar=False, sync_dist=True)
            self.log(f"val_level{i}_acc", acc, prog_bar=False, sync_dist=True)
            total_loss += loss.detach()
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self):
        # Step LR scheduler (configured as epoch-based LambdaLR)
        try:
            self.lr_schedulers().step()
        except Exception:
            pass
        # log lr
        opt = self.optimizers()
        self.log("learning_rate", opt.param_groups[0]["lr"], prog_bar=True, sync_dist=True)

# ---------------------------
# Helper finetune / trainer builder (similar to your previous finetune)
# ---------------------------
def finetune_deit3(trainer_module: pl.LightningModule, config: ScalaDistillContextDeit3, model_config: FlexModelConfig, conf_description: str):
    early_stopping = EarlyStopping(monitor='val_loss', patience=getattr(config, "patience", 20), mode='min', verbose=True)
    checkpoint_callback = ModelCheckpoint(dirpath=getattr(config, "checkpoint_dir", "./"), filename=f"{conf_description}_best_model", monitor='val_loss', mode='min', save_top_k=1)
    callbacks = [early_stopping, checkpoint_callback]

    logger = None
    if getattr(config, "wandb_project_name", None):
        logger = WandbLogger(project=config.wandb_project_name, name=f"{conf_description}_deit3", config=model_config.get_flat_dict(), save_dir=getattr(config, "log_path", "./"))
    else:
        logger = False

    ddp = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)
    pl_trainer = pl.Trainer(logger=logger, callbacks=callbacks, max_epochs=config.epochs, accelerator="gpu", devices="auto", strategy=ddp, precision=config.precision, log_every_n_steps=10)
    train_loader, val_loader, test_loader = config.loader_function()
    pl_trainer.fit(trainer_module, train_loader, val_loader)

    # load best model and test
    trained = trainer_module
    if pl_trainer.is_global_zero:
        trained = type(trainer_module).load_from_checkpoint(checkpoint_callback.best_model_path, training_context=config, model_config=model_config)
    pl_trainer.test(trained, dataloaders=test_loader, verbose=False)
    return trained