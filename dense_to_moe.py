"""
Dense -> Mixture-of-Experts (sparse upcycling), end to end.

Stages
  1. Train a small dense GPT (char-level) from scratch, log loss, save checkpoint.
  2. Convert it to an MoE: every FFN becomes N experts (copies of the dense FFN
     + small noise) with a linear router and top-k routing.
  3. Verify function preservation (MoE loss right after conversion ~= dense loss).
  4. Continue training the MoE (main loss + load-balancing aux loss), tracking
     expert utilization.
  5. Control: continue training the dense model for the same number of steps.
  6. Automatic PASS/FAIL checks that loss really goes down in every phase.
  7. Sample text from dense and MoE, save plots, JSON logs, checkpoints,
     summary.json and a human-readable report.md.

Usage
  python dense_to_moe.py                      # default run (GPU recommended, CPU ok)
  python dense_to_moe.py --quick              # tiny smoke test (~1 min on CPU)
  python dense_to_moe.py --dense_steps 3000 --moe_steps 3000 --n_experts 8 --top_k 2

Data: downloads Tiny Shakespeare; falls back to a synthetic corpus if offline.
"""
import argparse, copy, json, math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
p.add_argument("--out_dir", default="runs")
p.add_argument("--seed", type=int, default=1337)
# model
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--d_model", type=int, default=128)
p.add_argument("--d_ff", type=int, default=512)
p.add_argument("--block_size", type=int, default=128)
p.add_argument("--dropout", type=float, default=0.0)
# moe
p.add_argument("--n_experts", type=int, default=4)
p.add_argument("--top_k", type=int, default=2)
p.add_argument("--noise_std", type=float, default=0.01, help="relative noise added to expert copies")
p.add_argument("--aux_coef", type=float, default=0.01, help="load-balancing loss weight")
# training
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--dense_steps", type=int, default=2000)
p.add_argument("--moe_steps", type=int, default=2000)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--moe_lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--eval_every", type=int, default=100)
p.add_argument("--eval_iters", type=int, default=20)
args = p.parse_args()

if args.quick:
    args.dense_steps, args.moe_steps = 150, 150
    args.eval_every, args.eval_iters = 25, 5
    args.batch_size, args.warmup = 32, 20

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(args.out_dir, exist_ok=True)
print(f"device={device}")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_text():
    path = os.path.join(args.out_dir, "input.txt")
    if not os.path.exists(path):
        try:
            print("downloading Tiny Shakespeare ...")
            urllib.request.urlretrieve(URL, path)
        except Exception as e:
            print(f"download failed ({e}); using synthetic corpus")
            import random
            random.seed(0)
            words = ("the a of and to in is that it was he for on are with as his they be at one have this "
                     "from or had by hot but some what there we can out other were all your when up use word "
                     "how said an each she which do their time if will way about many then them write would "
                     "like so these her long make thing see him two has look more day could go come did number "
                     "sound no most people my over know water than call first who may down side been now find").split()
            text = " ".join(random.choice(words) for _ in range(400_000))
            with open(path, "w") as f:
                f.write(text)
    return open(path, encoding="utf-8").read()

text = load_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"corpus chars={len(text):,} vocab={vocab_size} train={len(train_data):,} val={len(val_data):,}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i:i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])
    return x.to(device), y.to(device)

# Fixed evaluation batches: every model/phase is scored on the SAME batches, so
# differences between dense / MoE / control are real, not sampling noise.
def make_eval_sets():
    sets = {}
    for split, d in (("train", train_data), ("val", val_data)):
        g = torch.Generator().manual_seed(999)
        ix = torch.randint(len(d) - args.block_size - 1, (args.eval_iters, args.batch_size), generator=g)
        sets[split] = [(torch.stack([d[i:i + args.block_size] for i in row]).to(device),
                        torch.stack([d[i + 1:i + 1 + args.block_size] for i in row]).to(device)) for row in ix]
    return sets
EVAL_SETS = make_eval_sets()

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.h, s.qkv, s.proj, s.drop = h, nn.Linear(d, 3 * d), nn.Linear(d, d), drop

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.drop if s.training else 0.0)
        return s.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class FFN(nn.Module):
    """Dense feed-forward block (the thing we upcycle)."""
    def __init__(s, d, d_ff):
        super().__init__()
        s.fc1, s.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(s, x):
        return s.fc2(F.gelu(s.fc1(x)))

class MoEFFN(nn.Module):
    """Top-k routed mixture of FFN experts with Switch-style load-balancing loss."""
    def __init__(s, d, d_ff, n_experts, top_k):
        super().__init__()
        s.E, s.k = n_experts, top_k
        s.router = nn.Linear(d, n_experts, bias=False)
        s.experts = nn.ModuleList([FFN(d, d_ff) for _ in range(n_experts)])
        s.aux_loss = torch.zeros(())
        s.last_counts = None  # token counts per expert (for utilization tracking)

    def forward(s, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        probs = F.softmax(s.router(xf).float(), dim=-1)            # (N, E)
        topv, topi = probs.topk(s.k, dim=-1)                       # (N, k)
        topv = topv / topv.sum(-1, keepdim=True)                   # renormalise -> weights sum to 1
        # (function-preserving: identical experts + weights summing to 1 == dense FFN)

        out = torch.zeros_like(xf)
        for e in range(s.E):
            tok, slot = (topi == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, s.experts[e](xf[tok]) * topv[tok, slot].unsqueeze(-1).to(xf.dtype))

        # load balancing: E * sum_e (fraction of assignments to e) * (mean router prob of e)
        counts = F.one_hot(topi, s.E).sum(1).float()               # (N, E)
        frac = counts.mean(0) / s.k
        s.aux_loss = s.E * (frac * probs.mean(0)).sum()
        s.last_counts = counts.sum(0).detach()
        return out.view(B, T, C)

class Block(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        s.attn = Attention(cfg.d_model, cfg.n_head, cfg.dropout)
        s.ffn = FFN(cfg.d_model, cfg.d_ff)

    def forward(s, x):
        x = x + s.attn(s.ln1(x))
        return x + s.ffn(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, cfg, vocab):
        super().__init__()
        s.cfg = cfg
        s.tok, s.pos = nn.Embedding(vocab, cfg.d_model), nn.Embedding(cfg.block_size, cfg.d_model)
        s.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        s.ln_f = nn.LayerNorm(cfg.d_model)
        s.head = nn.Linear(cfg.d_model, vocab, bias=False)
        s.head.weight = s.tok.weight  # weight tying
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def moe_layers(s):
        return [b.ffn for b in s.blocks if isinstance(b.ffn, MoEFFN)]

    def forward(s, idx, targets=None):
        B, T = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks:
            x = b(x)
        logits = s.head(s.ln_f(x))
        if targets is None:
            return logits, None, torch.zeros((), device=idx.device)
        main = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        moes = s.moe_layers()
        aux = torch.stack([m.aux_loss for m in moes]).mean() if moes else torch.zeros((), device=idx.device)
        return logits, main, aux

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    moes = model.moe_layers()
    if not moes:
        return total, total
    per_expert = sum(p.numel() for p in moes[0].experts[0].parameters())
    inactive = len(moes) * (moes[0].E - moes[0].k) * per_expert
    return total, total - inactive

# --------------------------------------------------------------------------- #
# Upcycling: dense -> MoE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def upcycle(dense: GPT, n_experts, top_k, noise_std):
    cfg = dense.cfg
    moe = copy.deepcopy(dense)
    for blk_d, blk_m in zip(dense.blocks, moe.blocks):
        layer = MoEFFN(cfg.d_model, cfg.d_ff, n_experts, top_k).to(next(dense.parameters()).device)
        for ex in layer.experts:
            ex.load_state_dict(blk_d.ffn.state_dict())              # copy dense weights
            if noise_std > 0:                                       # break symmetry
                for prm in ex.parameters():
                    prm.add_(torch.randn_like(prm) * noise_std * prm.std())
        nn.init.normal_(layer.router.weight, std=0.02)              # small router init
        blk_m.ffn = layer
    return moe

# --------------------------------------------------------------------------- #
# Train / eval helpers
# --------------------------------------------------------------------------- #
def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

@torch.no_grad()
def evaluate(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        ls = []
        for x, y in EVAL_SETS[split]:
            ls.append(model(x, y)[1].item())
        out[split] = sum(ls) / len(ls)
    model.train()
    return out

def utilization(model):
    """Per-layer fraction of token-assignments going to each expert (from last fwd pass)."""
    res = []
    for m in model.moe_layers():
        c = m.last_counts
        res.append((c / c.sum()).tolist() if c is not None else None)
    return res

@torch.no_grad()
def generate(model, prompt="\n", n=300, temperature=0.8):
    itos = {i: c for c, i in stoi.items()}
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    for _ in range(n):
        logits = model(idx[:, -args.block_size:])[0][:, -1] / temperature
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    model.train()
    return "".join(itos[i] for i in idx[0].tolist())

def loss_check(name, log, min_drop=0.01):
    """PASS if the final val/train loss of a phase is lower than its first eval."""
    ev = [r for r in log if "val_loss" in r]
    a, b = ev[0], ev[-1]
    return dict(phase=name, start_val=a["val_loss"], end_val=b["val_loss"], start_train=a["train_loss"],
                end_train=b["train_loss"], val_drop=a["val_loss"] - b["val_loss"],
                train_drop=a["train_loss"] - b["train_loss"],
                passed=bool(a["val_loss"] - b["val_loss"] > min_drop and a["train_loss"] - b["train_loss"] > min_drop))

def run_phase(name, model, steps, base_lr, log, step_offset, aux_coef=0.0):
    decay = [q for q in model.parameters() if q.dim() >= 2]      # matrices only
    no_decay = [q for q in model.parameters() if q.dim() < 2]    # biases / LayerNorm
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)],
                            lr=base_lr, betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    for step in range(steps + 1):
        gstep = step_offset + step
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, base_lr, args.warmup)
        if step % args.eval_every == 0 or step == steps:
            ev = evaluate(model)
            rec = dict(phase=name, step=gstep, train_loss=ev["train"], val_loss=ev["val"])
            if model.moe_layers():
                rec["utilization"] = utilization(model)
            log.append(rec)
            print(f"[{name}] step {gstep:5d} | train {ev['train']:.4f} | val {ev['val']:.4f} | {time.time()-t0:.0f}s")
        if step == steps:
            break
        x, y = get_batch("train")
        _, main, aux = model(x, y)
        loss = main + aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            log.append(dict(phase=name, step=gstep, kind="batch", total_loss=loss.item(),
                            main_loss=main.item(), aux_loss=float(aux.detach())))
    return log

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = args
    dense = GPT(cfg, vocab_size).to(device)
    d_total, _ = count_params(dense)
    print(f"dense params: {d_total/1e6:.2f}M")

    # ---- Stage 1: dense training ----
    dense_log = []
    run_phase("dense", dense, args.dense_steps, args.lr, dense_log, 0)
    torch.save(dense.state_dict(), os.path.join(args.out_dir, "dense.pt"))
    dense_ckpt_loss = evaluate(dense)

    # ---- Stage 2: conversion ----
    # Fixed eval batches so the before/after comparison is exactly apples-to-apples.
    g = torch.Generator().manual_seed(123)
    fixed = [(val_data[i:i + args.block_size], val_data[i + 1:i + 1 + args.block_size])
             for i in torch.randint(len(val_data) - args.block_size - 1, (256,), generator=g).tolist()]
    fx = torch.stack([a for a, _ in fixed]).to(device)
    fy = torch.stack([b for _, b in fixed]).to(device)

    @torch.no_grad()
    def fixed_loss(m):
        m.eval()
        l = m(fx, fy)[1].item()
        m.train()
        return l

    dense_fixed = fixed_loss(dense)
    moe = upcycle(dense, args.n_experts, args.top_k, 0.0)           # noise=0 -> exact check
    moe_fixed_exact = fixed_loss(moe)
    print(f"\nfunction-preservation check (noise=0): dense={dense_fixed:.6f} moe={moe_fixed_exact:.6f} "
          f"diff={abs(dense_fixed-moe_fixed_exact):.2e}")
    moe = upcycle(dense, args.n_experts, args.top_k, args.noise_std)  # real init with noise
    moe_fixed_noisy = fixed_loss(moe)
    print(f"with noise_std={args.noise_std}: moe loss right after conversion = {moe_fixed_noisy:.6f}")
    m_total, m_active = count_params(moe)
    print(f"MoE params: total {m_total/1e6:.2f}M | active/token {m_active/1e6:.2f}M\n")

    # ---- Stage 3: MoE training + dense control ----
    moe_log = []
    run_phase("moe", moe, args.moe_steps, args.moe_lr, moe_log, args.dense_steps, aux_coef=args.aux_coef)
    ctrl = GPT(cfg, vocab_size).to(device)
    ctrl.load_state_dict(torch.load(os.path.join(args.out_dir, "dense.pt"), map_location=device))
    ctrl_log = []
    run_phase("dense_continued", ctrl, args.moe_steps, args.moe_lr, ctrl_log, args.dense_steps)
    torch.save(moe.state_dict(), os.path.join(args.out_dir, "moe.pt"))

    # ---- Verification: does loss actually go down? ----
    checks = [loss_check("dense (from scratch)", dense_log),
              loss_check("MoE (after conversion)", moe_log),
              loss_check("dense continued (control)", ctrl_log)]
    evals = lambda L: [r for r in L if "val_loss" in r]
    moe_last = evals(moe_log)[-1]
    checks.append(dict(phase="MoE vs dense-at-conversion", start_val=dense_ckpt_loss["val"],
                       end_val=moe_last["val_loss"], val_drop=dense_ckpt_loss["val"] - moe_last["val_loss"],
                       passed=bool(moe_last["val_loss"] < dense_ckpt_loss["val"])))
    for c in checks:
        print(f"[check] {'PASS' if c['passed'] else 'FAIL'} | {c['phase']:32s} val {c['start_val']:.4f} -> {c['end_val']:.4f}")
    samples = dict(dense=generate(dense), moe=generate(moe))
    with open(os.path.join(args.out_dir, "samples.txt"), "w") as f:
        f.write("=== DENSE ===\n" + samples["dense"] + "\n\n=== MoE ===\n" + samples["moe"] + "\n")

    # ---- Summaries ----
    final = dict(
        dense_at_conversion=dense_ckpt_loss,
        moe_final=evals(moe_log)[-1], dense_continued_final=evals(ctrl_log)[-1],
        function_preservation=dict(dense=dense_fixed, moe_exact=moe_fixed_exact, moe_with_noise=moe_fixed_noisy),
        params=dict(dense=d_total, moe_total=m_total, moe_active=m_active),
        config={k: v for k, v in vars(args).items()},
        final_expert_utilization=utilization(moe),
        loss_reduction_checks=checks,
        all_checks_passed=all(c["passed"] for c in checks),
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(args.out_dir, "logs.json"), "w") as f:
        json.dump(dict(dense=dense_log, moe=moe_log, dense_continued=ctrl_log), f)

    print("\n=== SUMMARY (val loss) ===")
    print(f"dense @ conversion       : {dense_ckpt_loss['val']:.4f}")
    print(f"MoE after {args.moe_steps} steps       : {final['moe_final']['val_loss']:.4f}")
    print(f"dense continued (control): {final['dense_continued_final']['val_loss']:.4f}")
    write_report(final, checks)
    plot(dense_log, moe_log, ctrl_log, final)

def write_report(final, checks):
    fp, pr = final["function_preservation"], final["params"]
    L = ["# Dense -> MoE report", "",
         f"- Dense: {args.n_layer} layers, d_model={args.d_model}, d_ff={args.d_ff}, {pr['dense']/1e6:.2f}M params",
         f"- MoE: {args.n_experts} experts, top-{args.top_k}, {pr['moe_total']/1e6:.2f}M total / {pr['moe_active']/1e6:.2f}M active per token",
         f"- Data: char-level, {len(text):,} chars, vocab {vocab_size}",
         f"- Steps: dense {args.dense_steps}, MoE {args.moe_steps} (+ dense-continued control)", "",
         "## Function preservation at conversion (fixed val batches)",
         f"dense {fp['dense']:.6f} | MoE noise=0 {fp['moe_exact']:.6f} | MoE noise={args.noise_std} {fp['moe_with_noise']:.6f}", "",
         "## Loss-reduction checks", "", "| phase | start val | end val | drop | result |", "|---|---|---|---|---|"]
    for c in checks:
        L.append(f"| {c['phase']} | {c['start_val']:.4f} | {c['end_val']:.4f} | {c['val_drop']:.4f} | {'PASS' if c['passed'] else 'FAIL'} |")
    L += ["", f"**All checks passed: {final['all_checks_passed']}**", "",
          "## Final val loss", f"- dense @ conversion: {final['dense_at_conversion']['val']:.4f}",
          f"- MoE final: {final['moe_final']['val_loss']:.4f}",
          f"- dense continued (control): {final['dense_continued_final']['val_loss']:.4f}", "",
          "Plots: loss_curves.png | logs: logs.json | samples: samples.txt"]
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")

def plot(dense_log, moe_log, ctrl_log, final):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    ev = lambda L, k: ([r["step"] for r in L if k in r and "val_loss" in r], [r[k] for r in L if k in r and "val_loss" in r])
    conv = args.dense_steps
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    for L, lab, c in ((dense_log, "dense", "C0"), (moe_log, "MoE (upcycled)", "C3"), (ctrl_log, "dense continued", "C2")):
        xs, ys = ev(L, "val_loss"); a.plot(xs, ys, label=f"{lab} val", color=c)
        xs, ys = ev(L, "train_loss"); a.plot(xs, ys, "--", alpha=.5, color=c)
    a.axvline(conv, color="k", ls=":", label="conversion")
    a.set(title="Train (dashed) / val (solid) loss", xlabel="step", ylabel="cross-entropy"); a.legend()

    a = ax[0, 1]
    b = [r for r in moe_log if r.get("kind") == "batch"]
    a.plot([r["step"] for r in b], [r["main_loss"] for r in b], label="main")
    a.plot([r["step"] for r in b], [r["total_loss"] for r in b], label="total (main + aux)")
    a.set(title="MoE batch losses", xlabel="step"); a.legend()
    a2 = a.twinx(); a2.plot([r["step"] for r in b], [r["aux_loss"] for r in b], "g", alpha=.5)
    a2.set_ylabel("aux (green; 1.0 = balanced)")

    a = ax[1, 0]
    ue = [r for r in moe_log if "utilization" in r and r["utilization"][0] is not None]
    for e in range(args.n_experts):
        a.plot([r["step"] for r in ue], [sum(l[e] for l in r["utilization"]) / len(r["utilization"]) for r in ue], label=f"expert {e}")
    a.axhline(1 / args.n_experts, color="k", ls=":")
    a.set(title="Expert utilization (mean over layers)", xlabel="step", ylabel="token fraction"); a.legend()

    a = ax[1, 1]
    fp = final["function_preservation"]
    names = ["dense", "MoE exact\n(noise=0)", f"MoE noise\n={args.noise_std}"]
    vals = [fp["dense"], fp["moe_exact"], fp["moe_with_noise"]]
    a.bar(names, vals, color=["C0", "C3", "C1"])
    a.set_ylim(min(vals) * 0.98, max(vals) * 1.02)
    a.set(title="Loss right at conversion (fixed batches)", ylabel="loss")
    plt.tight_layout()
    path = os.path.join(args.out_dir, "loss_curves.png")
    plt.savefig(path, dpi=130)
    print(f"saved {path}")

if __name__ == "__main__":
    main()
