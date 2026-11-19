"""
Reorders the hidden channels and MLP neurons of a Hugging Face LLaMA-family model
(Qwen3-1.7B) by activation importance, so that the leading slices taken by the
smaller FlexLLaMA levels hold the most important ones (Minitron-style width
importance). The permutation does not change the full model's output.

Importance is measured on medmix training text:
  - hidden channel: mean |output| of every RMSNorm (both per-layer norms and the
    final norm), summed over layers; one permutation for the whole residual stream;
  - MLP neuron: mean |input| of down_proj, one permutation per layer;
  - attention head (--heads): mean L2 norm of the head's output (the input of
    o_proj), one permutation per layer. Heads move in GQA groups, a kv head with
    the query heads that use it, so the query-to-kv pairing is kept; the groups
    are sorted by the summed importance of their query heads, and the query heads
    within a group by their own importance.

Afterwards, without any training, compares the LM loss of every level of the
FlexLLaMA built from the original and from the reordered weights, for each of
the given level configs (by default without and with keep_kv_group).

    python reorder_qwen3.py                      # writes hf_export/qwen3_1p7b_reordered
    python reorder_qwen3.py --heads              # writes hf_export/qwen3_1p7b_reordered_heads
"""
import argparse
import dataclasses
import json
import math
import os
import random

import torch
import torch.nn.functional as F


def read_texts(path: str, n: int, seed: int):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return [r["prompt"] + r["answer"] for r in rows]


def batches(tokenizer, texts, batch_size: int, max_len: int, device):
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len, add_special_tokens=False)
        yield enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def importance(model, tokenizer, texts, batch_size: int, max_len: int):
    """(hidden importance [H], MLP importance per layer [L, I], head importance per layer [L, heads])."""
    layers = model.model.layers
    hidden = torch.zeros(model.config.hidden_size, device=model.device, dtype=torch.float64)
    mlp = torch.zeros(len(layers), model.config.intermediate_size, device=model.device, dtype=torch.float64)
    n_heads = model.config.num_attention_heads
    heads = torch.zeros(len(layers), n_heads, device=model.device, dtype=torch.float64)
    mask = {}

    def mean_abs(t):
        m = mask["m"].unsqueeze(-1).to(t.dtype)
        return (t.abs() * m).sum((0, 1)).double() / m.sum()

    # Hooks must return None: a returned value would replace the module's output.
    def on_norm(_, __, out):
        hidden.add_(mean_abs(out))

    def on_down_proj(i):
        def hook(_, inp, __):
            mlp[i].add_(mean_abs(inp[0]))
        return hook

    def on_o_proj(i):
        def hook(_, inp, __):
            x = inp[0]
            per_head = x.reshape(*x.shape[:2], n_heads, -1).float().norm(dim=-1)  # [B, S, heads]
            m = mask["m"].unsqueeze(-1).to(per_head.dtype)
            heads[i].add_((per_head * m).sum((0, 1)).double() / m.sum())
        return hook

    hooks = [model.model.norm.register_forward_hook(on_norm)]
    for i, layer in enumerate(layers):
        hooks.append(layer.input_layernorm.register_forward_hook(on_norm))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(on_norm))
        hooks.append(layer.mlp.down_proj.register_forward_hook(on_down_proj(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(on_o_proj(i)))
    for input_ids, attention_mask in batches(tokenizer, texts, batch_size, max_len, model.device):
        mask["m"] = attention_mask
        model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    return hidden, mlp, heads


def permute_state_dict(sd: dict, num_layers: int, perm_h, perm_mlp, tied: bool) -> dict:
    sd = {k: v.clone() for k, v in sd.items()}
    sd["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"][:, perm_h]
    sd["model.norm.weight"] = sd["model.norm.weight"][perm_h]
    for i in range(num_layers):
        p, pm = f"model.layers.{i}", perm_mlp[i]
        for norm in ("input_layernorm", "post_attention_layernorm"):
            sd[f"{p}.{norm}.weight"] = sd[f"{p}.{norm}.weight"][perm_h]
        for proj in ("q_proj", "k_proj", "v_proj"):
            sd[f"{p}.self_attn.{proj}.weight"] = sd[f"{p}.self_attn.{proj}.weight"][:, perm_h]
        sd[f"{p}.self_attn.o_proj.weight"] = sd[f"{p}.self_attn.o_proj.weight"][perm_h, :]
        for proj in ("gate_proj", "up_proj"):
            sd[f"{p}.mlp.{proj}.weight"] = sd[f"{p}.mlp.{proj}.weight"][pm][:, perm_h]
        sd[f"{p}.mlp.down_proj.weight"] = sd[f"{p}.mlp.down_proj.weight"][perm_h][:, pm]
    if tied:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    else:
        sd["lm_head.weight"] = sd["lm_head.weight"][:, perm_h]
    return sd


def head_order(head_imp, num_kv_heads: int):
    """Query-head order of one layer: GQA groups by summed importance, then the
    query heads within each group by their own importance. Returns (query head
    order, kv head order)."""
    group = len(head_imp) // num_kv_heads
    per_group = head_imp.reshape(num_kv_heads, group)
    kv_order = torch.argsort(per_group.sum(-1), descending=True).tolist()
    q_order = [g * group + r for g in kv_order
               for r in torch.argsort(per_group[g], descending=True).tolist()]
    return q_order, kv_order


def permute_heads(sd: dict, num_layers: int, head_dim: int, orders) -> dict:
    """Applies per-layer (query head order, kv head order) to q/k/v/o_proj."""
    sd = dict(sd)

    def rows(order):
        return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in order])

    for i in range(num_layers):
        p = f"model.layers.{i}.self_attn"
        q_rows, kv_rows = rows(orders[i][0]), rows(orders[i][1])
        for name, idx in ((f"{p}.q_proj", q_rows), (f"{p}.k_proj", kv_rows), (f"{p}.v_proj", kv_rows)):
            sd[f"{name}.weight"] = sd[f"{name}.weight"][idx]
            if f"{name}.bias" in sd:
                sd[f"{name}.bias"] = sd[f"{name}.bias"][idx]
        sd[f"{p}.o_proj.weight"] = sd[f"{p}.o_proj.weight"][:, q_rows]
    return sd


def reorder(model, tokenizer, texts, batch_size: int, max_len: int, heads: bool = False):
    """Reorders `model` in place; returns the hidden and MLP permutations."""
    hidden, mlp, head_imp = importance(model, tokenizer, texts, batch_size, max_len)
    perm_h = torch.argsort(hidden, descending=True).cpu()
    perm_mlp = [torch.argsort(m, descending=True).cpu() for m in mlp]
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    num_layers = len(model.model.layers)
    sd = permute_state_dict(sd, num_layers, perm_h, perm_mlp, tied)
    if heads:
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        orders = [head_order(h.cpu(), cfg.num_key_value_heads) for h in head_imp]
        sd = permute_heads(sd, num_layers, head_dim, orders)
    model.load_state_dict(sd)
    return perm_h, perm_mlp


@torch.no_grad()
def level_losses(model, tokenizer, texts, batch_size: int, max_len: int):
    """Mean next-token loss over real tokens, per level."""
    out = []
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        total, count = 0.0, 0
        for input_ids, attention_mask in batches(tokenizer, texts, batch_size, max_len, next(model.parameters()).device):
            logits = model(input_ids)[:, :-1].float()
            keep = attention_mask[:, 1:].bool()
            total += F.cross_entropy(logits[keep], input_ids[:, 1:][keep], reduction="sum").item()
            count += int(keep.sum())
        out.append(total / count)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out", default=None,
                        help="default hf_export/qwen3_1p7b_reordered, with --heads ..._reordered_heads")
    parser.add_argument("--heads", action="store_true", help="also reorder the attention heads")
    parser.add_argument("--force", action="store_true", help="reorder again even if --out exists")
    parser.add_argument("--calib", default=os.path.join("data", "medmix_sft", "train.jsonl"))
    parser.add_argument("--val", default=os.path.join("data", "medmix_sft", "val.jsonl"))
    parser.add_argument("--n_calib", type=int, default=512)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--configs", default=None,
                        help="semicolon-separated configs whose levels are compared")
    args = parser.parse_args()
    reordered = os.path.join("hf_export", "qwen3_1p7b_reordered")
    if args.out is None:
        args.out = reordered + ("_heads" if args.heads else "")
    if args.configs is None:
        args.configs = ("flexllama,sft.medmix_qwen3_1p7b_10L_kv_teacher" if args.heads else
                        "flexllama,sft.medmix_qwen3_1p7b_10L_shard;flexllama,sft.medmix_qwen3_1p7b_10L_kv_shard")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import utils
    from run_experiment import resolve_from_str

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    val = read_texts(args.val, args.n_val, seed=1)

    if os.path.isdir(args.out) and not args.force:
        print(f"using the existing {args.out} (--force reorders again)")
    else:
        original = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.float32).to(device).eval()
        model = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.float32).to(device).eval()
        calib = read_texts(args.calib, args.n_calib, seed=0)
        reorder(model, tokenizer, calib, args.batch_size, args.max_len, heads=args.heads)

        # The reordered model must compute the same function as the original. A
        # different summation order alone leaves float32 rounding of ~1e-2 on logits
        # of size ~20, so compare what the models predict: the loss and the top token.
        # A wrong permutation changes both by far more than these bounds.
        with torch.no_grad():
            worst, same, count, loss_a, loss_b = 0.0, 0, 0, 0.0, 0.0
            for input_ids, attention_mask in batches(tokenizer, val[:16], args.batch_size, args.max_len, device):
                a = original(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
                b = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
                keep = attention_mask[:, 1:].bool()
                targets = input_ids[:, 1:][keep]
                worst = max(worst, (a - b)[keep].abs().max().item())
                same += int((a[keep].argmax(-1) == b[keep].argmax(-1)).sum())
                count += int(keep.sum())
                loss_a += F.cross_entropy(a[keep], targets, reduction="sum").item()
                loss_b += F.cross_entropy(b[keep], targets, reduction="sum").item()
        loss_a, loss_b, agree = loss_a / count, loss_b / count, same / count
        print(f"reordered vs original: loss {loss_a:.5f} vs {loss_b:.5f}, same top token {100 * agree:.2f}%, "
              f"max |logit diff| {worst:.2e}")
        if abs(loss_a - loss_b) > 1e-3 or agree < 0.995:
            raise SystemExit("MISMATCH: the reordered model is not equivalent, nothing written")
        del original

        # The released Qwen3 weights are bf16; reindexing them loses nothing.
        model.to(torch.bfloat16).save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        print(f"saved {args.out}")
        del model

    print(f"\nLM loss on {len(val)} medmix val examples, no training (lower is better; "
          f"a uniform guess is {math.log(tokenizer.vocab_size):.2f})")
    sources = [("original", args.source), ("reordered", args.out)]
    if args.out != reordered and os.path.isdir(reordered):
        sources = [("original", args.source), ("reord", reordered), ("reord+heads", args.out)]
    for conf in args.configs.split(";"):
        cfg = resolve_from_str(conf).model_config
        losses = {}
        for name, src in sources:
            flex = dataclasses.replace(cfg, pretrained_hf_model=src).make_model()
            utils.load_llama_weights_into_flexllama(flex, src)
            flex.to(device).eval()
            losses[name] = level_losses(flex, tokenizer, val, args.batch_size, args.max_len)
            del flex
        print(f"\n{conf}  (kv_group {getattr(cfg, 'kv_group', None)})")
        print(f"{'level':>5} {'hidden':>6} {'heads/kv':>8} " + " ".join(f"{n:>11}" for n, _ in sources))
        for level in range(len(cfg.hidden_dims)):
            heads, kv = list(cfg.num_heads)[level], list(cfg.num_kv_heads)[level]
            print(f"{level:>5} {list(cfg.hidden_dims)[level]:>6} {f'{heads}/{kv}':>8} "
                  + " ".join(f"{losses[n][level]:11.3f}" for n, _ in sources))


if __name__ == "__main__":
    main()
