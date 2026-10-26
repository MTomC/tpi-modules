"""
experiments.py — orquestación de los experimentos controlados (Pasos 1.2, 2.3 y proxy de QA).

Idea central: el modelo se corre UNA vez por imagen (run_owl_raw -> caché en disco) y todo lo demás
(threshold, NMS, modo de etiquetas, tope de objetos) se barre offline en segundos con
`sweep_detection`. Para decidir parámetros usar las 120 imágenes de development, no 5 ni 20.
"""
from __future__ import annotations

import itertools
import json
from typing import Callable

import numpy as np

import dataset_utils as du
import vision_pipeline as vp
import query_engine as qe


def make_grid(**options) -> list[dict]:
    """make_grid(score_thr=[.1,.2], nms_mode=['global','batched']) -> lista de dicts (producto cartesiano)."""
    keys = list(options)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(options[k] for k in keys))]


def dets_to_preds(dets: list[dict], vocab_ids: list[int]) -> list[dict]:
    return [{"category_id": int(vocab_ids[d["label"]]), "score": d["score"], "bbox": d["bbox"]} for d in dets]


def sweep_detection(raw_by_img: dict[int, dict], dev: du.DevData, vocab_ids: list[int],
                    grid: list[dict], iou_thresh: float = 0.5):
    """Paso 1.2: una fila por combinación de parámetros con F1 (micro y macro), TP/FP/FN y precisión/recall."""
    import pandas as pd
    gt_by_img = {i: dev.gt_for_image(i) for i in raw_by_img}
    rows = []
    for params in grid:
        preds_by_img = {}
        for i, raw in raw_by_img.items():
            info = dev.images[i]
            dets = vp.decode_owl_outputs(raw["logits"], raw["boxes"], int(info["width"]), int(info["height"]), **params)
            preds_by_img[i] = dets_to_preds(dets, vocab_ids)
        r = du.evaluate_dataset(preds_by_img, gt_by_img, iou_thresh)
        rows.append({**params, "TP": r["tp"], "FP": r["fp"], "FN": r["fn"], "precision": round(r["precision"], 4),
                     "recall": round(r["recall"], 4), "F1_micro": round(r["f1_micro"], 4),
                     "F1_macro": round(r["f1_macro"], 4), "ignored_preds": r["ignored_preds"],
                     "n_preds": sum(len(v) for v in preds_by_img.values())})
    return pd.DataFrame(rows).sort_values("F1_micro", ascending=False).reset_index(drop=True)


def evaluate_with_masks(raw_by_img: dict[int, dict], dev: du.DevData, vocab_ids: list[int], params: dict,
                        image_loader: Callable[[int], np.ndarray], segment_fn: Callable[[np.ndarray, list], list],
                        bbox_source: str = "owl", iou_thresh: float = 0.5) -> dict:
    """
    Paso 2.3: con los parámetros elegidos en 1.2, segmenta y calcula F1 de cajas Y de máscaras.
    bbox_source='mask' reemplaza la caja por la caja ajustada a la máscara (experimento: ¿acerca la caja al GT?).
    segment_fn(image_np, boxes_xyxy) -> lista de máscaras (H,W); p.ej. perception.segment.
    """
    gt_boxes = {i: dev.gt_for_image(i) for i in raw_by_img}
    gt_masks = {i: dev.gt_for_image(i, with_masks=True) for i in raw_by_img}
    preds_by_img = {}
    for i, raw in raw_by_img.items():
        info = dev.images[i]
        w, h = int(info["width"]), int(info["height"])
        dets = vp.decode_owl_outputs(raw["logits"], raw["boxes"], w, h, **params)
        masks = segment_fn(image_loader(i), [d["bbox"] for d in dets]) if dets else []
        preds = []
        for d, m in zip(dets, masks):
            box = (vp.mask_tight_bbox(m) or d["bbox"]) if bbox_source == "mask" else d["bbox"]
            preds.append({"category_id": int(vocab_ids[d["label"]]), "score": d["score"], "bbox": box, "mask": m})
        preds_by_img[i] = preds
    box = du.evaluate_dataset(preds_by_img, gt_boxes, iou_thresh, "box")
    msk = du.evaluate_dataset(preds_by_img, gt_masks, iou_thresh, "mask")
    return {"box_f1": box["f1_micro"], "mask_f1": msk["f1_micro"], "box": {k: box[k] for k in ("tp", "fp", "fn")},
            "mask": {k: msk[k] for k in ("tp", "fp", "fn")}, "preds_by_img": preds_by_img,
            "box_per_image": box["per_image"], "mask_per_image": msk["per_image"]}


def plot_gt_vs_pred(dev: du.DevData, images_dir, preds_by_img: dict[int, list[dict]], image_ids: list[int],
                    show_masks: bool = False) -> None:
    """Prueba aislada: GT en verde, predicciones en rojo (cajas y, opcionalmente, contornos de máscara)."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    for i in image_ids:
        gt = dev.gt_for_image(i, with_masks=show_masks)
        with Image.open(Path(images_dir) / gt["image"]["file_name"]).convert("RGB") as im:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.imshow(im)
            for a in gt["anns"]:
                x1, y1, x2, y2 = a["bbox_xyxy"]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.4))
                if show_masks and a["mask_np"].any():
                    ax.contour(a["mask_np"], levels=[0.5], colors="lime", linewidths=1)
            for p in preds_by_img.get(i, []):
                x1, y1, x2, y2 = p["bbox"]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1, ls="--"))
                ax.text(x1, y2, f"{dev.categories.get(p['category_id'], p['category_id'])} {p['score']:.2f}", fontsize=6,
                        color="white", bbox=dict(fc="red", alpha=.6, pad=1))
                if show_masks and "mask" in p and np.asarray(p["mask"]).any():
                    ax.contour(np.asarray(p["mask"]), levels=[0.5], colors="red", linewidths=0.8)
            ax.set_title(f"{gt['image']['file_name']}  (verde=GT, rojo=pred)")
            ax.axis("off")
            plt.show()
            plt.close(fig)


# ----------------------------------------------------------------------
# QA: GT proxy y exactitud
# ----------------------------------------------------------------------
def scene_from_gt(dev: du.DevData, image_id: int) -> dict:
    """SceneJSON construido con el GT (confianza 1.0): sirve para derivar las respuestas esperadas."""
    gt = dev.gt_for_image(image_id, with_masks=True)
    info = gt["image"]
    dets = [{"category_id": a["category_id"], "bbox": a["bbox_xyxy"], "score": 1.0, "mask": a["mask_np"]} for a in gt["anns"]]
    return vp.build_scene(dev.image_key(image_id), int(info["width"]), int(info["height"]), dets)


def expected_answers(dev: du.DevData, qa_rows: list[dict], engine: qe.QueryEngine) -> dict[str, object]:
    """Respuesta 'esperada' = el mismo motor aplicado al GT. Es un PROXY: la verdad de Kaggle no se publica."""
    scenes = {}
    out = {}
    for q in qa_rows:
        if q["type"] == "description":
            continue
        i = int(q["image_id"])
        scenes.setdefault(i, scene_from_gt(dev, i))
        out[q["question_id"]] = engine.answer(scenes[i], q)["normalized_answer"]
    return out


def qa_accuracy(scenes_by_image: dict[int, dict], qa_rows: list[dict], engine: qe.QueryEngine,
                expected: dict[str, object]):
    """Exactitud global y por tipo. Con ~40 preguntas, cada una vale 2.5 pts: no sobre-optimizar."""
    import pandas as pd
    rows = []
    for q in qa_rows:
        if q["question_id"] not in expected:
            continue
        pred = engine.answer(scenes_by_image[int(q["image_id"])], q)["normalized_answer"]
        rows.append({"question_id": q["question_id"], "type": q["type"], "pred": pred,
                     "expected": expected[q["question_id"]], "ok": pred == expected[q["question_id"]]})
    df = pd.DataFrame(rows)
    return df, (float(df["ok"].mean()) if len(df) else 0.0), df.groupby("type")["ok"].agg(["mean", "count"])


def score_submission_df(sub_df, dev: du.DevData, kind: str = "box", iou_thresh: float = 0.5, strict: bool = False) -> dict:
    """
    Puntúa con reglas LVIS las filas de DEVELOPMENT de un submission (columnas image_id, SceneJSON).
    Reemplaza el bloque de métricas de instancias de `auto_evaluate_metrics` del notebook v27, que cuenta como
    FP las predicciones de categorías no verificadas (README: deben ignorarse).
    """
    key_to_id = {dev.image_key(i): i for i in dev.images}
    preds_by_img, gt_by_img = {}, {}
    for _, row in sub_df.iterrows():
        i = key_to_id.get(str(row["image_id"]))
        if i is None:
            continue
        scene = json.loads(row["SceneJSON"])
        preds = []
        for o in scene["objects"]:
            p = {"category_id": int(o["category_id"]), "score": float(o["confidence"]), "bbox": o["bbox"]}
            if kind == "mask":
                p["mask"] = vp.rle_to_mask(o["mask"])
            preds.append(p)
        preds_by_img[i] = preds
        gt_by_img[i] = dev.gt_for_image(i, with_masks=(kind == "mask"))
    if not gt_by_img:
        raise ValueError("el submission no tiene filas de development (ids tipo 'tpi_0005')")
    return du.evaluate_dataset(preds_by_img, gt_by_img, iou_thresh, kind, strict)
