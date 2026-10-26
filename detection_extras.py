"""
detection_extras.py — piezas OPCIONALES para subir F1 de cajas/máscaras. Medir cada una con experiments.py
antes de adoptarla; ninguna mejora está garantizada.

PROBADO sin GPU (ver test_extras.py):  weighted_box_fusion, tile_windows, shift_dets, federated_report, refine_mask
NO PROBADO (necesita GPU/pesos):       detect_tiled, GroundingDinoDetector

IMPORTANTE sobre 'LVIS federated': neg_category_ids / not_exhaustive_category_ids existen SOLO en las anotaciones
de development. En test no están, así que NO pueden usarse como filtro dentro del pipeline (sería inyectar
anotaciones). Solo sirven para el EVALUADOR local -> federated_report.
"""
from __future__ import annotations

import numpy as np

import dataset_utils as du
import vision_pipeline as vp


# ----------------------------------------------------------------------
# Weighted Box Fusion (WBF) — fusiona detecciones de varios modelos / teselas
# ----------------------------------------------------------------------
def weighted_box_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr: float = 0.55,
                        skip_thr: float = 0.0):
    """
    boxes_list[m] (N_m,4) xyxy en píxeles, scores_list[m] (N_m,), labels_list[m] (N_m,) enteros.
    Cada cluster (misma etiqueta, IoU > iou_thr con la caja fusionada) promedia coordenadas ponderadas por score*peso.
    Score fusionado = media de scores del cluster * min(n_cajas, n_modelos)/n_modelos
    (una caja que solo vio un modelo se penaliza).
    """
    n_models = len(boxes_list)
    weights = np.ones(n_models) if weights is None else np.asarray(weights, dtype=float)
    items = []
    for m in range(n_models):
        for b, s, l in zip(np.asarray(boxes_list[m], float).reshape(-1, 4), np.asarray(scores_list[m], float),
                           np.asarray(labels_list[m], int)):
            if s >= skip_thr:
                items.append((float(s * weights[m]), float(s), b, int(l), m))
    items.sort(key=lambda t: -t[0])
    clusters: list[dict] = []
    for ws, s, b, l, m in items:
        best, best_iou = None, iou_thr
        for c in clusters:
            if c["label"] != l:
                continue
            iou = float(du.box_iou_matrix([c["fused"]], [b])[0, 0])
            if iou > best_iou:
                best, best_iou = c, iou
        if best is None:
            clusters.append({"label": l, "boxes": [b], "ws": [ws], "scores": [s], "models": {m}, "fused": b.copy()})
        else:
            best["boxes"].append(b); best["ws"].append(ws); best["scores"].append(s); best["models"].add(m)
            w_arr = np.asarray(best["ws"])
            best["fused"] = (np.asarray(best["boxes"]) * w_arr[:, None]).sum(0) / w_arr.sum()
    out_b, out_s, out_l = [], [], []
    for c in clusters:
        out_b.append(c["fused"])
        out_s.append(float(np.mean(c["scores"]) * min(len(c["boxes"]), n_models) / n_models))
        out_l.append(c["label"])
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    order = np.argsort(-np.asarray(out_s))
    return np.asarray(out_b)[order], np.asarray(out_s)[order], np.asarray(out_l, int)[order]


def fuse_detections(dets_per_source: list[list[dict]], weights=None, iou_thr: float = 0.55,
                    max_per_image: int = 100) -> list[dict]:
    """dets: [{bbox, score, label}] por fuente (modelo o tesela) -> lista fusionada con el mismo formato."""
    b, s, l = weighted_box_fusion([[d["bbox"] for d in ds] or np.zeros((0, 4)) for ds in dets_per_source],
                                  [[d["score"] for d in ds] for ds in dets_per_source],
                                  [[d["label"] for d in ds] for ds in dets_per_source], weights, iou_thr)
    return [{"bbox": [float(v) for v in bb], "score": float(ss), "label": int(ll)}
            for bb, ss, ll in list(zip(b, s, l))[:max_per_image]]


# ----------------------------------------------------------------------
# Tiling estilo SAHI
# ----------------------------------------------------------------------
def tile_windows(w: int, h: int, rows: int = 2, cols: int = 2, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Ventanas (x0,y0,x1,y1) que cubren la imagen con solapamiento relativo `overlap`."""
    tw = int(np.ceil(w / (cols - (cols - 1) * overlap)))
    th = int(np.ceil(h / (rows - (rows - 1) * overlap)))
    xs = [0] if cols == 1 else [int(round(i * (w - tw) / (cols - 1))) for i in range(cols)]
    ys = [0] if rows == 1 else [int(round(i * (h - th) / (rows - 1))) for i in range(rows)]
    return [(max(0, x), max(0, y), min(w, x + tw), min(h, y + th)) for y in ys for x in xs]


def shift_dets(dets: list[dict], x0: int, y0: int) -> list[dict]:
    return [{**d, "bbox": [d["bbox"][0] + x0, d["bbox"][1] + y0, d["bbox"][2] + x0, d["bbox"][3] + y0]} for d in dets]


def detect_tiled(perception, image_pil, vocab_names, template, decode_params: dict,
                 rows: int = 2, cols: int = 2, overlap: float = 0.2, fuse_iou: float = 0.55) -> list[dict]:
    """[NO PROBADO, GPU] detección global + por teselas, fusionada con WBF. Cuesta (1 + rows*cols) pasadas de OWLv2."""
    w, h = image_pil.size
    sources = []
    raw = perception.run_owl_raw(image_pil, vocab_names, template)
    sources.append(vp.decode_owl_outputs(raw["logits"], raw["boxes"], w, h, **decode_params))
    for (x0, y0, x1, y1) in tile_windows(w, h, rows, cols, overlap):
        crop = image_pil.crop((x0, y0, x1, y1))
        r = perception.run_owl_raw(crop, vocab_names, template)
        sources.append(shift_dets(vp.decode_owl_outputs(r["logits"], r["boxes"], x1 - x0, y1 - y0, **decode_params), x0, y0))
    return fuse_detections(sources, iou_thr=fuse_iou, max_per_image=decode_params.get("max_per_image", 100))


# ----------------------------------------------------------------------
# Diagnóstico LVIS: ¿de dónde vienen los FP que SÍ cuentan?
# ----------------------------------------------------------------------
def federated_report(preds_by_img: dict, gt_by_img: dict, category_names: dict[int, str],
                     iou_thresh: float = 0.5, kind: str = "box"):
    """
    Por categoría: TP, FP que cuentan (negativas verificadas + exhaustivas), FN, predicciones ignoradas
    (categorías no verificadas / no exhaustivas) y soporte GT. La suma de TP/FP/FN coincide con evaluate_dataset.
    """
    import pandas as pd
    acc: dict[int, dict] = {}
    for key, gt in gt_by_img.items():
        preds = preds_by_img.get(key, [])
        cats = ({int(a["category_id"]) for a in gt["anns"]} | gt["neg"] | {int(p["category_id"]) for p in preds})
        for c in cats:
            gt_c = {"anns": [a for a in gt["anns"] if int(a["category_id"]) == c],
                    "neg": gt["neg"] & {c}, "not_exh": gt["not_exh"] & {c}, "image": gt.get("image", {})}
            pr_c = [p for p in preds if int(p["category_id"]) == c]
            r = du.evaluate_image(pr_c, gt_c, iou_thresh, kind)
            row = acc.setdefault(c, {"TP": 0, "FP_counted": 0, "FN": 0, "preds_ignored": 0, "GT": 0, "n_preds": 0})
            row["TP"] += r["tp"]; row["FP_counted"] += r["fp"]; row["FN"] += r["fn"]
            row["preds_ignored"] += r["ignored_preds"]; row["GT"] += len(gt_c["anns"]); row["n_preds"] += len(pr_c)
    df = pd.DataFrame.from_dict(acc, orient="index")
    df.insert(0, "category", [category_names.get(i, str(i)) for i in df.index])
    return df.sort_values(["FP_counted", "FN"], ascending=False)


# ----------------------------------------------------------------------
# Refinamiento de máscaras (morfología + componente principal)
# ----------------------------------------------------------------------
def refine_mask(mask: np.ndarray, box=None, open_k: int = 3, close_k: int = 3, keep_largest: bool = True,
                min_keep_ratio: float = 0.5) -> np.ndarray:
    """
    Apertura (quita motas) -> cierre (tapa grietas finas) -> conserva el componente conectado más grande.
    No rellena agujeros (anillos, aros). Si el resultado pierde más de la mitad del área, devuelve la original.
    `box` (x1,y1,x2,y2): lo que caiga fuera de la caja se descarta.
    """
    from scipy import ndimage as ndi
    m = np.asarray(mask) > 0
    area0 = int(m.sum())
    if area0 == 0:
        return m.astype(np.uint8)
    out = m.copy()
    if box is not None:
        h, w = m.shape
        x1, y1, x2, y2 = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
        clip = np.zeros_like(out); clip[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        out &= clip
    if open_k > 1:
        out = ndi.binary_opening(out, structure=np.ones((open_k, open_k), bool))
    if close_k > 1:
        out = ndi.binary_closing(out, structure=np.ones((close_k, close_k), bool))
    if keep_largest and out.any():
        lab, n = ndi.label(out)
        if n > 1:
            sizes = ndi.sum(out, lab, index=range(1, n + 1))
            out = lab == (1 + int(np.argmax(sizes)))
    return (out if out.sum() >= min_keep_ratio * area0 else m).astype(np.uint8)


# ----------------------------------------------------------------------
# Grounding DINO (NO PROBADO): esqueleto con las partes frágiles marcadas
# ----------------------------------------------------------------------
class GroundingDinoDetector:
    """
    Devuelve [{bbox, score, label(índice en vocab_names)}] para fusionar con OWLv2 vía fuse_detections.
    Puntos frágiles (verificar con 2 imágenes antes de correr todo):
      * la API de post_process_grounded_object_detection cambió entre versiones de transformers
        (threshold vs box_threshold, text_labels vs labels);
      * el prompt admite ~256 tokens: se parte el vocabulario en bloques de `chunk` categorías;
      * la etiqueta devuelta es texto libre y puede ser una concatenación de frases: se mapea por coincidencia
        exacta / aproximada y, si no hay match claro, la detección se descarta.
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: str | None = None, chunk: int = 20):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.chunk = chunk

    def detect(self, image_pil, vocab_names: list[str], box_thr: float = 0.25, text_thr: float = 0.25) -> list[dict]:
        import difflib
        w, h = image_pil.size
        lower = [n.lower() for n in vocab_names]
        dets = []
        for s in range(0, len(vocab_names), self.chunk):
            names = lower[s:s + self.chunk]
            text = " ".join(f"{n}." for n in names)
            inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = self.model(**inputs)
            try:
                res = self.processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[(h, w)])[0]
            except TypeError:
                res = self.processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=box_thr, text_threshold=text_thr, target_sizes=[(h, w)])[0]
            labels = res.get("text_labels", res.get("labels"))
            for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
                lab = str(lab).strip().lower()
                if lab in names:
                    idx = s + names.index(lab)
                else:
                    m = difflib.get_close_matches(lab, names, n=1, cutoff=0.85)
                    if not m:
                        continue
                    idx = s + names.index(m[0])
                bb = vp.clip_box(box.tolist(), w, h)
                if bb:
                    dets.append({"bbox": bb, "score": float(score), "label": idx})
        return dets
