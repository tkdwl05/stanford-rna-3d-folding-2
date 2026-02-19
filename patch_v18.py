"""
v18 → v19 추가 패치:
1. conf_logits NaN: _fail_if_nonfinite 이전에 conf_logits도 nan_to_num으로 sanitize
2. 진짜 원인: preds에 nonfinite → run_epoch에서 preds/conf_logits도 체크 후 skip
3. AMP 환경에서 conf_logits FP16 오버플로우 방지: ConfidenceHead forward에서 clamp
4. EGNN coord_update 폭발 방지: agg_dx를 clamp (coord_step_scale 말고 추가 clamp)
5. _fail_if_nonfinite: preds nonfinite도 skip 조건으로 추가 (loss 이외에도)
6. cfg.amp = False 로 고정 (FP16 자체를 끄는 게 가장 안전)
"""
import json, re

SRC = "improve_data_label_v18.ipynb"
DST = "improve_data_label_v19.ipynb"

with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

def cell_src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    lines = text.split("\n")
    result = [l + "\n" for l in lines]
    if result and result[-1] == "\n":
        result[-1] = ""
    cell["source"] = result

patched = 0

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell_src(cell)

    # ── FIX A: AMP를 끔 (FP16 오버플로우가 근본 원인) ──────────────────────
    OLD_AMP = "    amp: bool = True             # ✅ 필요하면 False로 꺼서 NaN 여부 확인"
    NEW_AMP = "    amp: bool = False            # ✅ v19: FP16 conf_logits/preds overflow 방지 → AMP OFF"
    if OLD_AMP in src:
        src = src.replace(OLD_AMP, NEW_AMP)
        print("✅ FIX A: amp=False (AMP 비활성화)")
        patched += 1

    # ── FIX B: run_epoch - preds/conf_logits도 nonfinite면 skip ─────────────
    OLD_SKIP = '''\
            # ✅ v18: non-finite loss → skip step (log warning) instead of hard crash
            if not torch.isfinite(loss):
                print(f"[NaN/skip] epoch={epoch} step={step} train={train}  loss={loss.item()}  stage={stage_name} — skipping batch")
                if train:
                    opt.zero_grad(set_to_none=True)
                continue'''
    NEW_SKIP = '''\
            # ✅ v19: non-finite loss OR preds OR conf_logits → skip step
            _preds_ok = torch.isfinite(preds).all()
            _conf_ok  = (conf_logits is None) or torch.isfinite(conf_logits).all()
            _loss_ok  = torch.isfinite(loss)
            if not (_loss_ok and _preds_ok and _conf_ok):
                print(f"[NaN/skip] epoch={epoch} step={step} train={train}  "
                      f"loss_ok={bool(_loss_ok.item())}  preds_ok={bool(_preds_ok.item())}  conf_ok={bool(_conf_ok.item() if hasattr(_conf_ok,'item') else _conf_ok)}  "
                      f"stage={stage_name} — skipping batch")
                if train:
                    opt.zero_grad(set_to_none=True)
                continue'''
    if OLD_SKIP in src:
        src = src.replace(OLD_SKIP, NEW_SKIP)
        print("✅ FIX B: run_epoch preds/conf도 nonfinite skip")
        patched += 1

    # ── FIX C: ConfidenceHead.forward → output clamp & nan_to_num ──────────
    OLD_CONF_FWD = '''\
    def forward(self, h, pad_mask):
        if pad_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = pad_mask.float().unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (h * m).sum(dim=1) / denom
        return self.mlp(pooled)  # (B,K)'''
    NEW_CONF_FWD = '''\
    def forward(self, h, pad_mask):
        if pad_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = pad_mask.float().unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (h * m).sum(dim=1) / denom
        out = self.mlp(pooled)  # (B,K)
        # ✅ v19: AMP FP16에서 overflow → NaN/Inf 방지
        out = torch.nan_to_num(out.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)
        return out'''
    if OLD_CONF_FWD in src:
        src = src.replace(OLD_CONF_FWD, NEW_CONF_FWD)
        print("✅ FIX C: ConfidenceHead output nan_to_num + clamp")
        patched += 1

    # ── FIX D: EGNNv16.forward - preds nan_to_num guard 추가 ────────────────
    OLD_PREDS_MASK = '''\
        preds = preds.masked_fill(~pad_mask[:, None, :, None], 0.0)

        conf_logits = self.conf_head(h, pad_mask)  # (B,K)
        return preds, conf_logits'''
    NEW_PREDS_MASK = '''\
        preds = preds.masked_fill(~pad_mask[:, None, :, None], 0.0)
        # ✅ v19: EGNN 누적 폭발 방지 (agg_dx index_add_ overflow)
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1e4, neginf=-1e4)
        preds = preds.clamp(-1e4, 1e4)

        conf_logits = self.conf_head(h, pad_mask)  # (B,K)
        return preds, conf_logits'''
    if OLD_PREDS_MASK in src:
        src = src.replace(OLD_PREDS_MASK, NEW_PREDS_MASK)
        print("✅ FIX D: EGNNv16 preds nan_to_num + clamp")
        patched += 1

    # ── FIX E: EGNNPairAwareLayer - agg_dx clamp ────────────────────────────
    OLD_AGGDX = '''\
        if node_mask is not None:
            agg_dx = agg_dx.masked_fill(~node_mask[:, :, None], 0.0)
            agg_m  = agg_m.masked_fill(~node_mask[:, :, None], 0.0)

        x = x + agg_dx'''
    NEW_AGGDX = '''\
        if node_mask is not None:
            agg_dx = agg_dx.masked_fill(~node_mask[:, :, None], 0.0)
            agg_m  = agg_m.masked_fill(~node_mask[:, :, None], 0.0)

        # ✅ v19: coord 폭발 방지 clamp (coord_step_scale 이후에도 누적될 수 있음)
        agg_dx = agg_dx.clamp(-5.0, 5.0)
        x = x + agg_dx'''
    if OLD_AGGDX in src:
        src = src.replace(OLD_AGGDX, NEW_AGGDX)
        print("✅ FIX E: EGNNPairAwareLayer agg_dx clamp(-5,5)")
        patched += 1

    # ── FIX F: cfg 헤더 주석 업데이트 ───────────────────────────────────────
    src = src.replace("[v18]", "[v19]").replace("[v16]", "[v19]")

    set_src(cell, src)

print(f"\nTotal patches: {patched}/5")

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"💾 Saved → {DST}")
