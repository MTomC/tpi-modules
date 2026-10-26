"""
audit_metric_crosscheck.py — ¿Las dos implementaciones del F1 que hay en el notebook dan lo mismo
entre sí y con el evaluador unificado de dataset_utils?

Uso:  python audit_metric_crosscheck.py /ruta/al/notebook.ipynb
Extrae `evaluate_lvis_image` (celda del benchmark) y `evaluate_box_f1` (celda de Fase 0) del .ipynb,
las ejecuta tal cual (con un shim numpy para torch.tensor / ops.box_iou) y las compara con
dataset_utils.evaluate_image sobre miles de casos aleatorios.
"""
import json
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dataset_utils as du


def load_original_functions(nb_path):
    cells = json.load(open(nb_path))["cells"]
    src_all = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]

    # --- evaluate_lvis_image (benchmark) ---
    src = next(s for s in src_all if "def evaluate_lvis_image" in s)
    start = src.index("def compute_mask_iou_matrix")
    end = src.index("# =====", src.index("def evaluate_lvis_image"))
    code = src[start:end]

    class _Arr:                                            # imita tensor.numpy()
        def __init__(self, a): self.a = a
        def numpy(self): return self.a

    def _tensor(x, dtype=None):
        return np.array(x, dtype=np.float32)

    def _box_iou(a, b):
        return _Arr(du.box_iou_matrix(a, b).astype(np.float32))

    ns = {"np": np, "torch": types.SimpleNamespace(tensor=_tensor, float32=None),
          "ops": types.SimpleNamespace(box_iou=_box_iou)}
    exec(code, ns)
    eval_lvis = ns["evaluate_lvis_image"]

    # --- evaluate_box_f1 (Fase 0) ---
    src12 = next(s for s in src_all if "def evaluate_box_f1" in s)
    code12 = src12[:src12.index("# Test 1")]
    ns12 = {"defaultdict": defaultdict, "np": np}
    exec(code12, ns12)
    # xywh_to_xyxy vive en otra celda: la tomamos del código del notebook también
    src9 = next(s for s in src_all if "def xywh_to_xyxy" in s)
    m = re.search(r"def xywh_to_xyxy.*?return \[x, y, x \+ w, y \+ h\]", src9, re.S)
    exec(m.group(0), ns12)
    return eval_lvis, ns12["evaluate_box_f1"]


def random_case(rng, with_masks=False, H=60, W=80):
    cats = list(range(1, 7))
    gt_cats = rng.choice(cats, size=rng.integers(1, 4), replace=False).tolist()
    rest = [c for c in cats if c not in gt_cats]
    neg = rng.choice(rest, size=rng.integers(0, len(rest) + 1), replace=False).tolist() if rest else []
    not_exh = [c for c in gt_cats if rng.random() < 0.3]

    def rbox():
        x1, y1 = rng.uniform(0, W - 12), rng.uniform(0, H - 12)
        return [x1, y1, x1 + rng.uniform(5, 30), y1 + rng.uniform(5, 30)]

    def clip(b): return [min(b[0], W - 1), min(b[1], H - 1), min(max(b[2], b[0] + 1), W), min(max(b[3], b[1] + 1), H)]

    gts = [{"category_id": int(c), "box": clip(rbox())} for c in gt_cats for _ in range(rng.integers(1, 4))]
    preds = []
    for g in gts:                                            # preds ruidosas alrededor del GT
        for _ in range(rng.integers(0, 3)):
            jit = rng.normal(0, 3, 4)
            preds.append({"category_id": g["category_id"], "box": clip([v + j for v, j in zip(g["box"], jit)]),
                          "score": float(rng.random())})
    for _ in range(rng.integers(0, 6)):                      # preds espurias en cualquier categoría
        preds.append({"category_id": int(rng.choice(cats)), "box": clip(rbox()), "score": float(rng.random())})

    def mk_mask(b):
        m = np.zeros((H, W), np.uint8)
        m[int(b[1]):int(np.ceil(b[3])), int(b[0]):int(np.ceil(b[2]))] = 1
        return m
    return gts, preds, neg, not_exh, mk_mask, (H, W)


def main(nb_path, n=3000):
    eval_lvis, eval_box_f1 = load_original_functions(nb_path)
    rng = np.random.default_rng(0)
    diff_a = diff_b = diff_mask = 0
    first_diff = None
    for t in range(n):
        gts, preds, neg, not_exh, mk_mask, (H, W) = random_case(rng)
        # formato unificado
        info = {"neg_category_ids": neg, "not_exhaustive_category_ids": not_exh, "width": W, "height": H}
        anns = [{"category_id": g["category_id"], "bbox": [g["box"][0], g["box"][1], g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]]} for g in gts]
        gt_new = du.prepare_gt(info, anns)
        preds_new = [{"category_id": p["category_id"], "bbox": p["box"], "score": p["score"]} for p in preds]
        new = du.evaluate_image(preds_new, gt_new)
        new_t = (new["tp"], new["fp"], new["fn"])

        # original A (celda benchmark)
        gt_a = [{"category_id": g["category_id"], "bbox_xyxy": g["box"]} for g in gts]
        pr_a = [{"category_id": p["category_id"], "bbox": p["box"], "score": p["score"]} for p in preds]
        a = eval_lvis(pr_a, gt_a, set(neg), set(not_exh), 0.5, eval_masks=False)

        # original B (celda Fase 0)
        gt_b = {"img": {"image": info, "annotations": anns}}
        b = eval_box_f1({"img": preds_new}, gt_b)
        b_t = (b["tp"], b["fp"], b["fn"])

        if a != new_t:
            diff_a += 1; first_diff = first_diff or ("benchmark vs nuevo", a, new_t)
        if b_t != new_t:
            diff_b += 1; first_diff = first_diff or ("fase0 vs nuevo", b_t, new_t)

        # máscaras: benchmark original vs nuevo
        if t < 800:
            gt_am = [{"category_id": g["category_id"], "bbox_xyxy": g["box"], "mask_np": mk_mask(g["box"])} for g in gts]
            pr_am = [{"category_id": p["category_id"], "bbox": p["box"], "score": p["score"], "mask_np": mk_mask(p["box"])} for p in preds]
            am = eval_lvis(pr_am, gt_am, set(neg), set(not_exh), 0.5, eval_masks=True)
            gt_nm = du.prepare_gt(info, anns)
            for gg, gm in zip(gt_nm["anns"], gt_am):
                gg["mask_np"] = gm["mask_np"]
            pr_nm = [{**p, "mask": m["mask_np"]} for p, m in zip(preds_new, pr_am)]
            nm = du.evaluate_image(pr_nm, gt_nm, kind="mask")
            if am != (nm["tp"], nm["fp"], nm["fn"]):
                diff_mask += 1; first_diff = first_diff or ("mask benchmark vs nuevo", am, (nm["tp"], nm["fp"], nm["fn"]))

    print(f"casos: {n} | divergencias  benchmark(celda5) vs nuevo: {diff_a} | fase0(celda12) vs nuevo: {diff_b} "
          f"| máscaras benchmark vs nuevo (800 casos): {diff_mask}")
    if first_diff:
        print("primera divergencia:", first_diff)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/notebook0cb38a9804.ipynb")
