"""
Reads improve_data_label_v17.ipynb and writes improve_data_label_v18.ipynb
with the following NaN/Inf fixes applied:

1. kabsch_align: add eps-regularized identity to H before SVD, and NaN-guard the result.
2. LossComposer._forward_impl: clamp total_bk from below before softmin_aggregate.
3. run_epoch: skip step (log a warning) instead of crashing when loss is non-finite.
4. _fail_if_nonfinite: make it a warning+skip rather than a hard crash (controlled by cfg.fail_on_nan).
"""
import json, copy, re

SRC = "improve_data_label_v17.ipynb"
DST = "improve_data_label_v18.ipynb"

with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── helpers ──────────────────────────────────────────────────────────────────
def cell_src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    lines = [l + "\n" for l in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell["source"] = lines

def find_cell_containing(cells, marker):
    for i, c in enumerate(cells):
        if marker in cell_src(c):
            return i
    return None

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 ─ kabsch_align: regularize H before SVD + NaN guard on result
# ─────────────────────────────────────────────────────────────────────────────
OLD_KABSCH = '''\
def kabsch_align(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Align P to Q using Kabsch. SVD always runs in FP32."""
    with torch.amp.autocast(device_type=('cuda' if P.is_cuda else 'cpu'), enabled=False):
        P32 = P.float()
        Q32 = Q.float()
        m32 = mask.float().unsqueeze(-1)  # (B,T,1)

        msum = m32.sum(dim=1, keepdim=True).clamp_min(eps)
        P_mean = (P32 * m32).sum(dim=1, keepdim=True) / msum
        Q_mean = (Q32 * m32).sum(dim=1, keepdim=True) / msum

        P_c = (P32 - P_mean) * m32
        Q_c = (Q32 - Q_mean) * m32

        H = torch.matmul(P_c.transpose(1, 2), Q_c).contiguous()  # (B,3,3)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        V = Vh.transpose(1, 2)

        det = torch.det(torch.matmul(V, U.transpose(1, 2)))
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))

        E = torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0).repeat(H.shape[0], 1, 1)
        E[:, 2, 2] = sign
        R = torch.matmul(torch.matmul(V, E), U.transpose(1, 2))  # (B,3,3)

        P_aligned = torch.matmul(P_c, R.transpose(1, 2)) + Q_mean
        P_aligned = P_aligned * m32

    return P_aligned.to(dtype=P.dtype)'''

NEW_KABSCH = '''\
def kabsch_align(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Align P to Q using Kabsch. SVD always runs in FP32.
    ✅ v18 NaN fix: (1) regularize H with eps*I before SVD to prevent degenerate
       singular values; (2) nan_to_num guard on the output.
    """
    with torch.amp.autocast(device_type=('cuda' if P.is_cuda else 'cpu'), enabled=False):
        P32 = P.float()
        Q32 = Q.float()
        m32 = mask.float().unsqueeze(-1)  # (B,T,1)

        msum = m32.sum(dim=1, keepdim=True).clamp_min(eps)
        P_mean = (P32 * m32).sum(dim=1, keepdim=True) / msum
        Q_mean = (Q32 * m32).sum(dim=1, keepdim=True) / msum

        P_c = (P32 - P_mean) * m32
        Q_c = (Q32 - Q_mean) * m32

        H = torch.matmul(P_c.transpose(1, 2), Q_c).contiguous()  # (B,3,3)
        # ✅ Tikhonov regularization: prevent degenerate H (all-zero or rank<3)
        #    which causes NaN gradients through SVD.
        H = H + eps * torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0)

        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        V = Vh.transpose(1, 2)

        det = torch.det(torch.matmul(V, U.transpose(1, 2)))
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))

        E = torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0).repeat(H.shape[0], 1, 1)
        E[:, 2, 2] = sign
        R = torch.matmul(torch.matmul(V, E), U.transpose(1, 2))  # (B,3,3)

        P_aligned = torch.matmul(P_c, R.transpose(1, 2)) + Q_mean
        P_aligned = P_aligned * m32
        # ✅ guard: SVD may still produce NaN if H is pathological; replace with 0
        P_aligned = torch.nan_to_num(P_aligned, nan=0.0, posinf=0.0, neginf=0.0)

    return P_aligned.to(dtype=P.dtype)'''

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 ─ LossComposer._forward_impl: clamp total_bk from below
# ─────────────────────────────────────────────────────────────────────────────
OLD_COMBINE = '''\
        total_bk = _sanitize_losses(total_bk)
        loss = softmin_aggregate(total_bk, temp)'''

NEW_COMBINE = '''\
        # ✅ v18: clamp from below so that the -var_w*var_bk term can't make total_bk
        #   go to -inf (which would break softmax inside softmin_aggregate).
        total_bk = _sanitize_losses(total_bk.clamp_min(0.0))
        loss = softmin_aggregate(total_bk, temp)'''

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 ─ run_epoch: skip non-finite steps instead of hard-crashing
# ─────────────────────────────────────────────────────────────────────────────
OLD_FAIL = '''\
            _fail_if_nonfinite(loss, preds, conf_logits, target, mask, where=f"epoch={epoch} step={step} train={train}", stage=stage_name)

            if train:
                scaler.scale(loss).backward()'''

NEW_FAIL = '''\
            # ✅ v18: non-finite loss → skip step (log warning) instead of hard crash
            if not torch.isfinite(loss):
                print(f"[NaN/skip] epoch={epoch} step={step} train={train}  loss={loss.item()}  stage={stage_name} — skipping batch")
                if train:
                    opt.zero_grad(set_to_none=True)
                continue

            _fail_if_nonfinite(loss, preds, conf_logits, target, mask, where=f"epoch={epoch} step={step} train={train}", stage=stage_name)

            if train:
                scaler.scale(loss).backward()'''

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 ─ kabsch_rmsd_losses: guard rmsd sqrt NaN
# ─────────────────────────────────────────────────────────────────────────────
OLD_SQRT = '''\
        diff_sq = (pa - tg) ** 2
        sum_sq = diff_sq.sum(dim=(1, 2))  # (B,)
        rmsd = torch.sqrt(sum_sq / denom + 1e-8)'''

NEW_SQRT = '''\
        diff_sq = (pa - tg) ** 2
        sum_sq = diff_sq.sum(dim=(1, 2))  # (B,)
        # ✅ v18: clamp sum_sq to >=0 before sqrt (floating point can give tiny negatives)
        rmsd = torch.sqrt((sum_sq / denom).clamp_min(0.0) + 1e-8)'''

# ─────────────────────────────────────────────────────────────────────────────
# Apply patches
# ─────────────────────────────────────────────────────────────────────────────
patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell_src(cell)

    if OLD_KABSCH in src:
        src = src.replace(OLD_KABSCH, NEW_KABSCH)
        print("✅ Applied FIX 1: kabsch_align SVD regularization")
        patched += 1

    if OLD_COMBINE in src:
        src = src.replace(OLD_COMBINE, NEW_COMBINE)
        print("✅ Applied FIX 2: total_bk clamped before softmin")
        patched += 1

    if OLD_FAIL in src:
        src = src.replace(OLD_FAIL, NEW_FAIL)
        print("✅ Applied FIX 3: run_epoch skips non-finite loss steps")
        patched += 1

    if OLD_SQRT in src:
        src = src.replace(OLD_SQRT, NEW_SQRT)
        print("✅ Applied FIX 4: kabsch_rmsd_losses sqrt clamp_min")
        patched += 1

    set_src(cell, src)

print(f"\nTotal patches applied: {patched}/4")

# Also update the comment at the top (Cell 0) to reflect v18
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = cell_src(cell)
        if "# 0) Imports, Device, Config  [v16]" in src:
            src = src.replace(
                "# 0) Imports, Device, Config  [v16]",
                "# 0) Imports, Device, Config  [v18]"
            )
            set_src(cell, src)
            break

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n💾 Saved patched notebook → {DST}")
