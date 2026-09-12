"""Train the evaluation network from labelled positions.

    python tools/train.py data/pos_*.txt --hidden 256 --epochs 30 --out net.npz

Architecture, matching what fastnnue.py runs: 768 piece-square inputs seen
from each side, a shared layer of `hidden` units per side, SCReLU, and one
output over both sides concatenated with the side to move first.

Target: a blend of the teacher's score through a sigmoid and the game result,
both from the side to move's point of view. The loss is squared error in
win-probability space, which weights the near-equal positions that decide
most games more than the lopsided ones.

--compare <net.npz> scores an existing net on the same validation split, with
its output scale fitted, so two nets are judged on the same positions.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

PIECES = "PNBRQKpnbrqk"
PAD = 768
MAXP = 32


def featurize(lines):
    """FEN lines -> (stm, nstm) index arrays, padded, plus score and result."""
    n = len(lines)
    stm = np.full((n, MAXP), PAD, dtype=np.int16)
    nstm = np.full((n, MAXP), PAD, dtype=np.int16)
    score = np.zeros(n, dtype=np.float32)
    result = np.zeros(n, dtype=np.float32)
    plane_of = {c: i for i, c in enumerate(PIECES)}
    keep = np.ones(n, dtype=bool)
    for r, line in enumerate(lines):
        try:
            fen, cp, res = line.split(" | ")
            placement, turn = fen.split(" ", 2)[:2]
            black = turn == "b"
            k = 0
            rank, file = 7, 0
            for ch in placement:
                if ch == "/":
                    rank -= 1
                    file = 0
                elif ch.isdigit():
                    file += ord(ch) - 48
                else:
                    plane = plane_of[ch]
                    index = rank * 8 + file
                    flipped = plane + 6 if plane < 6 else plane - 6
                    white_row = plane * 64 + index
                    black_row = flipped * 64 + (index ^ 56)
                    if black:
                        stm[r, k], nstm[r, k] = black_row, white_row
                    else:
                        stm[r, k], nstm[r, k] = white_row, black_row
                    k += 1
                    file += 1
            score[r] = float(cp)
            rw = float(res)
            result[r] = 1.0 - rw if black else rw
        except Exception:
            keep[r] = False
    return stm[keep], nstm[keep], score[keep], result[keep]


def load(patterns, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["stm"], d["nstm"], d["score"], d["result"]
    files = sorted(f for p in patterns for f in glob.glob(p))
    parts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            lines = [l.rstrip("\n") for l in fh if l.count(" | ") == 2]
        parts.append(featurize(lines))
        print(f"{f}: {len(parts[-1][0])} positions", flush=True)
    out = [np.concatenate([p[i] for p in parts]) for i in range(4)]
    if cache:
        np.savez(cache, stm=out[0], nstm=out[1], score=out[2], result=out[3])
    return out


class Net(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(PAD + 1, hidden, mode="sum", padding_idx=PAD)
        self.bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(2 * hidden, 1)
        with torch.no_grad():
            self.ft.weight.normal_(0, 0.05)
            self.ft.weight[PAD].zero_()
            self.bias.fill_(0.1)
            self.out.weight.normal_(0, 0.02)
            self.out.bias.zero_()

    def forward(self, stm, nstm):
        a = self.ft(stm) + self.bias
        b = self.ft(nstm) + self.bias
        x = torch.cat([a, b], 1).clamp(0, 1).square()
        return self.out(x).squeeze(1)

    def clip(self, qa, qb):
        # Keep the quantised weights inside int16. The accumulator itself is
        # int32 in the search, so only the stored weights need the bound.
        with torch.no_grad():
            self.ft.weight.clamp_(-1.98, 1.98)
            self.out.weight.clamp_(-32000 / qb, 32000 / qb)


def export(net, path, qa, qb, scale):
    w0 = net.ft.weight.detach().numpy()[:PAD]
    b0 = net.bias.detach().numpy()
    ow = net.out.weight.detach().numpy()[0]
    ob = net.out.bias.detach().numpy()[0]
    np.savez(path,
             w0=np.round(w0 * qa).astype(np.int16),
             b0=np.round(b0 * qa).astype(np.int32),
             ow=np.round(ow * qb).astype(np.int16),
             ob=np.array([round(ob * qa * qb)], dtype=np.int32),
             hidden=np.array([w0.shape[1]], dtype=np.int32),
             qa=np.array([qa], dtype=np.int32),
             qb=np.array([qb], dtype=np.int32),
             scale=np.array([scale], dtype=np.int32),
             inputs=np.array([768], dtype=np.int32))


def quantised_logits(path, stm, nstm):
    """What fastnnue.py computes, before its centipawn scale, as a float."""
    d = np.load(path)
    qa, qb = int(d["qa"][0]), int(d["qb"][0])
    w0 = np.vstack([d["w0"].astype(np.int64), np.zeros((1, d["w0"].shape[1]), np.int64)])
    b0 = d["b0"].astype(np.int64)
    ow = d["ow"].astype(np.int64)
    ob = int(d["ob"][0])
    out = np.zeros(len(stm))
    for s in range(0, len(stm), 4096):
        a = w0[stm[s:s + 4096].astype(np.int64)].sum(1) + b0
        b = w0[nstm[s:s + 4096].astype(np.int64)].sum(1) + b0
        x = np.clip(np.concatenate([a, b], 1), 0, qa)
        total = (x * x) @ ow
        out[s:s + 4096] = (total // qa + ob) / (qa * qb)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+")
    p.add_argument("--cache")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lam", type=float, default=0.75, help="weight of the score vs the result")
    p.add_argument("--k", type=float, default=400.0, help="sigmoid scale for teacher cp")
    p.add_argument("--qa", type=int, default=255)
    p.add_argument("--qb", type=int, default=64)
    p.add_argument("--out", default="net.npz")
    p.add_argument("--compare", nargs="*", default=[])
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    stm, nstm, score, result = load(a.data, a.cache)
    n = len(stm)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    nval = min(100000, n // 50)
    val, tr = order[:nval], order[nval:]
    print(f"{n} positions, {len(tr)} train, {nval} validation", flush=True)

    target = (a.lam / (1 + np.exp(-score / a.k)) + (1 - a.lam) * result).astype(np.float32)
    T = torch.from_numpy
    stm_t, nstm_t, tgt_t = T(stm.astype(np.int64)), T(nstm.astype(np.int64)), T(target)
    score_val = score[val]

    def val_loss_from_logits(z):
        pred = 1 / (1 + np.exp(-z))
        return float(np.mean((pred - target[val]) ** 2))

    for path in a.compare:
        z = quantised_logits(path, stm[val], nstm[val])
        # Its own scale is unknown; fit the one that suits it best.
        best = min((val_loss_from_logits(z * s), s) for s in np.geomspace(0.05, 20, 200))
        corr = np.corrcoef(z, score_val)[0, 1]
        print(f"compare {path}: val loss {best[0]:.6f} at scale {best[1]:.3f}, "
              f"corr with teacher {corr:.4f}", flush=True)

    net = Net(a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    vs, vn, vt = stm_t[val], nstm_t[val], tgt_t[val]
    best_val = 1e9
    for epoch in range(a.epochs):
        t0 = time.time()
        perm = torch.from_numpy(tr[rng.permutation(len(tr))])
        total = 0.0
        batches = 0
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            pred = torch.sigmoid(net(stm_t[idx], nstm_t[idx]))
            loss = torch.mean((pred - tgt_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            net.clip(a.qa, a.qb)
            total += loss.item()
            batches += 1
        sched.step()
        with torch.no_grad():
            vl = float(torch.mean((torch.sigmoid(net(vs, vn)) - vt) ** 2))
        if vl < best_val:
            best_val = vl
            export(net, a.out, a.qa, a.qb, int(a.k))
        print(f"epoch {epoch:2d} train {total / batches:.6f} val {vl:.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    z = quantised_logits(a.out, stm[val], nstm[val])
    print(f"exported {a.out}: quantised val loss {val_loss_from_logits(z):.6f}, "
          f"corr with teacher {np.corrcoef(z, score_val)[0, 1]:.4f}")

    # Centipawns the search understands. Every margin in it was tuned
    # against the reference net's units (its logit x 110), so fit this net's
    # logit onto those over the non-decisive positions and store the factor.
    if a.compare:
        z_ref = quantised_logits(a.compare[0], stm[val], nstm[val]) * 110
        mask = np.abs(score_val) < 800
        cpscale = float(np.sum(z_ref[mask] * z[mask]) / np.sum(z[mask] ** 2))
        d = dict(np.load(a.out))
        d["cpscale"] = np.array([round(cpscale)], dtype=np.int32)
        np.savez(a.out, **d)
        print(f"cpscale {cpscale:.1f} written to {a.out}")


if __name__ == "__main__":
    main()
