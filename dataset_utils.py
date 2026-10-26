"""
dataset_utils.py  —  FASE 0: dataset, ground truth y evaluador.

Regla de aislamiento: este módulo NO importa torch / transformers / sam2.
Solo numpy (+ pandas / matplotlib / pycocotools de forma perezosa).

Contenido
---------
  1. Rutas y carga            (resolve_base, Paths, DevData, load_categories)
  2. GT                       (decode_gt_mask, prepare_gt)
  3. EDA                      (eda_report, audit_qa_questions, plot_gt_samples)
  4. Evaluador LVIS (UNICO)   (evaluate_image, evaluate_dataset, f1_from_counts)

El evaluador reemplaza a las dos versiones duplicadas del notebook
(`evaluate_lvis_image` de la celda 5 y `evaluate_box_f1` de la celda 12).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COMPETITION_SLUG = "tpi-ia-2026-primavera-comprension-visual-de-escenas"
_BASE_CANDIDATES = [
    Path("/kaggle/input/competitions") / COMPETITION_SLUG,
    Path("/kaggle/input") / COMPETITION_SLUG,
]


# ======================================================================
# 1. RUTAS Y CARGA
# ======================================================================
def resolve_base(explicit: str | Path | None = None) -> Path:
    """Devuelve la carpeta base de la competencia (prueba las dos rutas de Kaggle)."""
    candidates = ([Path(explicit)] if explicit else []) + _BASE_CANDIDATES
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró la carpeta de la competencia en: "
                            + ", ".join(str(c) for c in candidates))


@dataclass(frozen=True)
class Paths:
    base: Path
    annotations: Path
    dev_images: Path
    test_images: Path
    categories: Path
    qa_file: Path
    sample_submission: Path
    schema: Path

    @classmethod
    def from_base(cls, base: str | Path | None = None) -> "Paths":
        b = resolve_base(base)
        pub = b / "tpi_ia_2026_competition_public"
        return cls(
            base=b,
            annotations=b / "development" / "annotations.json",
            dev_images=b / "development" / "images",
            test_images=pub / "test_images",
            categories=pub / "categories.csv",
            qa_file=b / "tpi_2026_qa_public" / "questions_dev.jsonl",
            sample_submission=b / "sample_submission.csv",
            schema=b / "scene_schema.json",
        )


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}, línea {n}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n} no es un objeto JSON")
            rows.append(row)
    return rows


def load_categories(path: str | Path) -> dict[str, Any]:
    """
    Lee categories.csv. IMPORTANTE: el orden de `vocab_ids` / `vocab_names` es el
    orden de las consultas de texto de OWLv2 (índice de etiqueta -> category_id).
    """
    with Path(path).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vocab_ids = [int(r["category_id"]) for r in rows]
    vocab_names = [r["category_name"] for r in rows]
    if len(set(vocab_ids)) != len(vocab_ids):
        raise ValueError("category_id duplicados en categories.csv")
    return {
        "vocab_ids": vocab_ids,
        "vocab_names": vocab_names,
        "name_to_id": dict(zip(vocab_names, vocab_ids)),
        "id_to_name": dict(zip(vocab_ids, vocab_names)),
    }


class DevData:
    """Índices sobre development/annotations.json (formato COCO + metadata LVIS)."""

    def __init__(self, raw: dict):
        missing = {"images", "categories", "annotations"} - raw.keys()
        if missing:
            raise ValueError(f"Faltan claves en annotations.json: {sorted(missing)}")
        self.raw = raw
        self.images: dict[int, dict] = {int(r["id"]): r for r in raw["images"]}
        self.categories: dict[int, str] = {int(r["id"]): r["name"] for r in raw["categories"]}
        self.name_to_id = {v: k for k, v in self.categories.items()}
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in raw["annotations"]:
            self.anns_by_image[int(ann["image_id"])].append(ann)

    @classmethod
    def load(cls, path: str | Path) -> "DevData":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def image_key(self, image_id: int) -> str:
        """'tpi_0005' para la imagen 5 (nombre de archivo sin extensión)."""
        return Path(self.images[image_id]["file_name"]).stem

    def gt_for_image(self, image_id: int, with_masks: bool = False) -> dict:
        return prepare_gt(self.images[image_id], self.anns_by_image.get(image_id, []), with_masks)


# ======================================================================
# 2. GROUND TRUTH
# ======================================================================
def xywh_to_xyxy(bbox) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox debe tener 4 valores xywh: {bbox}")
    x, y, w, h = map(float, bbox)
    return [x, y, x + w, y + h]


def decode_gt_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """
    Máscara binaria (H, W) uint8 de una anotación COCO.
    OJO: en este dataset `segmentation` son POLÍGONOS (lista de listas), no RLE.
    El RLE solo existe en la SALIDA (SceneJSON). Si iscrowd=1 puede venir un dict RLE.
    """
    seg = ann["segmentation"]
    try:
        from pycocotools import mask as mu
        if isinstance(seg, list):                       # polígonos
            rle = mu.merge(mu.frPyObjects(seg, height, width))
        elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):  # RLE sin comprimir
            rle = mu.frPyObjects(seg, height, width)
        else:                                           # RLE comprimido
            rle = seg
        return mu.decode(rle).astype(np.uint8)
    except ImportError:
        from PIL import Image, ImageDraw
        if not isinstance(seg, list):
            raise
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
        return np.array(img, dtype=np.uint8)


def prepare_gt(image_info: dict, anns: list[dict], with_masks: bool = False) -> dict:
    """GT listo para el evaluador: bbox_xyxy, (mask_np) + reglas LVIS de la imagen."""
    h, w = int(image_info["height"]), int(image_info["width"])
    prepared = []
    for a in anns:
        c = dict(a)
        c["category_id"] = int(a["category_id"])
        c["bbox_xyxy"] = xywh_to_xyxy(a["bbox"])
        if with_masks:
            c["mask_np"] = decode_gt_mask(a, h, w)
        prepared.append(c)
    return {
        "image": image_info,
        "anns": prepared,
        "neg": {int(x) for x in image_info.get("neg_category_ids", [])},
        "not_exh": {int(x) for x in image_info.get("not_exhaustive_category_ids", [])},
    }


# ======================================================================
# 3. EDA
# ======================================================================
def eda_report(dev: DevData, size_small_max: float = 0.02, size_medium_max: float = 0.08) -> dict:
    """Paso 0.1. Devuelve un dict de DataFrames/Series/escalares (sin torch)."""
    import pandas as pd

    img_df = pd.DataFrame([{"image_id": i, "width": r["width"], "height": r["height"],
                            "n_neg": len(r.get("neg_category_ids", [])),
                            "n_not_exh": len(r.get("not_exhaustive_category_ids", []))}
                           for i, r in dev.images.items()])
    img_df["area_px"] = img_df["width"] * img_df["height"]
    ann_df = pd.DataFrame([{"image_id": int(a["image_id"]), "category_id": int(a["category_id"]),
                            "bbox_area": float(a.get("area") or a["bbox"][2] * a["bbox"][3]),
                            "iscrowd": int(a.get("iscrowd", 0)),
                            "seg_is_polygon": isinstance(a.get("segmentation"), list)}
                           for a in dev.raw["annotations"]])
    ann_df["category_name"] = ann_df["category_id"].map(dev.categories)
    ann_df = ann_df.merge(img_df[["image_id", "area_px"]], on="image_id", how="left")
    ann_df["area_ratio"] = (ann_df["bbox_area"] / ann_df["area_px"]).clip(upper=1.0)

    n_ann_per_img = ann_df.groupby("image_id").size().reindex(img_df["image_id"], fill_value=0)
    not_exh_cat = {(i, int(c)) for i, r in dev.images.items() for c in r.get("not_exhaustive_category_ids", [])}
    in_not_exh = ann_df.apply(lambda r: (r["image_id"], r["category_id"]) in not_exh_cat, axis=1)

    ratio = ann_df["area_ratio"]
    return {
        "n_images": len(img_df), "n_annotations": len(ann_df), "n_categories": len(dev.categories),
        "image_size": img_df[["width", "height", "area_px"]].describe().round(1),
        "images_without_annotations": int((n_ann_per_img == 0).sum()),
        "pct_images_without_annotations": float((n_ann_per_img == 0).mean()),
        "annotations_per_image": n_ann_per_img.describe().round(2),
        "top20_categories": ann_df["category_name"].value_counts().head(20),
        "bbox_area": ann_df["bbox_area"].describe().round(1),
        "area_ratio_quantiles": ratio.quantile([.05, .25, .5, .75, .95]).round(5),
        # Si estas proporciones están muy desbalanceadas, revisar los umbrales de 'size'.
        "size_class_share_with_current_thresholds": {
            "small": float((ratio < size_small_max).mean()),
            "medium": float(((ratio >= size_small_max) & (ratio < size_medium_max)).mean()),
            "large": float((ratio >= size_medium_max).mean()),
        },
        # Reglas LVIS: determinan qué se penaliza.
        "neg_ids_per_image": img_df["n_neg"].describe().round(2),
        "not_exhaustive_per_image": img_df["n_not_exh"].describe().round(2),
        "pct_annotations_in_not_exhaustive_categories": float(in_not_exh.mean()),
        "iscrowd_annotations": int(ann_df["iscrowd"].sum()),
        "pct_segmentation_polygon": float(ann_df["seg_is_polygon"].mean()),
        "images_df": img_df, "annotations_df": ann_df,
    }


def audit_qa_questions(dev: DevData, qa_rows: list[dict]):
    """
    Para cada pregunta: ¿cuántas instancias GT tiene cada clase y es positiva/negativa/no verificada?
    Sirve para saber si las preguntas ('la instancia de X') apuntan a instancias únicas y para
    reconstruir el GT de existence (yes = tiene GT; no = está en neg_category_ids).
    """
    import pandas as pd
    out = []
    for q in qa_rows:
        img_id = int(q["image_id"])
        info = dev.images[img_id]
        neg = {int(x) for x in info.get("neg_category_ids", [])}
        by_cat: dict[int, int] = defaultdict(int)
        for a in dev.anns_by_image.get(img_id, []):
            by_cat[int(a["category_id"])] += 1
        for name in q.get("class_en", []) or []:
            cid = dev.name_to_id.get(name)
            n = by_cat.get(cid, 0) if cid is not None else -1
            out.append({"question_id": q["question_id"], "type": q["type"], "image_id": img_id,
                        "class_en": name, "n_gt_instances": n,
                        "status": ("unknown_class" if cid is None else
                                   "positive" if n > 0 else "negative" if cid in neg else "unverified")})
    return pd.DataFrame(out)


def plot_gt_samples(dev: DevData, images_dir: str | Path, n: int = 5, seed: int = 42,
                    draw_masks: bool = True, save_dir: str | Path | None = None) -> list[int]:
    """Prueba aislada 0.1: n imágenes al azar con cajas (verde) y contornos de máscara GT."""
    import random
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = random.Random(seed)
    ids = sorted(i for i, r in dev.images.items() if (Path(images_dir) / r["file_name"]).exists())
    chosen = rng.sample(ids, min(n, len(ids)))
    for img_id in chosen:
        gt = dev.gt_for_image(img_id, with_masks=draw_masks)
        info = gt["image"]
        with Image.open(Path(images_dir) / info["file_name"]).convert("RGB") as im:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.imshow(im)
            for a in gt["anns"]:
                x1, y1, x2, y2 = a["bbox_xyxy"]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.2))
                ax.text(x1, y1, dev.categories[a["category_id"]], fontsize=6,
                        bbox=dict(fc="lime", alpha=.7, pad=1))
                if draw_masks and a["mask_np"].any():
                    ax.contour(a["mask_np"], levels=[0.5], colors="lime", linewidths=0.8)
            ax.set_title(f"{info['file_name']} | {len(gt['anns'])} anotaciones | "
                         f"neg={len(gt['neg'])} no_exh={len(gt['not_exh'])}")
            ax.axis("off")
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(save_dir) / f"gt_{img_id}.png", dpi=120, bbox_inches="tight")
            plt.show()
            plt.close(fig)
    return chosen


# ======================================================================
# 4. EVALUADOR LVIS (única implementación)
# ======================================================================
def box_iou_matrix(a, b) -> np.ndarray:
    """IoU (N, M) entre cajas xyxy."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def mask_iou_matrix(pred_masks, gt_masks) -> np.ndarray:
    """IoU (N, M) entre máscaras binarias (H, W). float32 es exacto hasta 16.7M píxeles."""
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)))
    P = np.stack([np.asarray(m).reshape(-1) > 0 for m in pred_masks]).astype(np.float32)
    G = np.stack([np.asarray(m).reshape(-1) > 0 for m in gt_masks]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)


def f1_from_counts(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def evaluate_image(preds: list[dict], gt: dict, iou_thresh: float = 0.5,
                   kind: str = "box", strict: bool = False) -> dict:
    """
    TP/FP/FN de UNA imagen con las reglas LVIS del README de development.

    preds : [{"category_id", "score", "bbox":[x1,y1,x2,y2], ("mask": ndarray HxW)}]
    gt    : salida de prepare_gt (anns con bbox_xyxy y, si kind='mask', mask_np)
    kind  : 'box' | 'mask'
    strict: False -> match si IoU >= umbral (LVIS/COCO); True -> IoU > umbral.

    Reglas:
      * categoría con GT            -> se evalúa (TP/FN; FP solo si NO es 'not_exhaustive')
      * categoría en neg_category_ids -> toda predicción es FP
      * cualquier otra categoría    -> las predicciones se IGNORAN (no penalizan)
    Matching: voraz por score descendente, cada pred toma el GT libre de mayor IoU.
    """
    if kind not in ("box", "mask"):
        raise ValueError("kind debe ser 'box' o 'mask'")
    gt_by_cat: dict[int, list[dict]] = defaultdict(list)
    for g in gt["anns"]:
        gt_by_cat[int(g["category_id"])].append(g)
    pred_by_cat: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pred_by_cat[int(p["category_id"])].append(p)

    neg, not_exh = gt["neg"], gt["not_exh"]
    tp = fp = fn = 0
    for cat in set(gt_by_cat) | neg:
        gts = gt_by_cat.get(cat, [])
        ps = sorted(pred_by_cat.get(cat, []), key=lambda x: -float(x["score"]))
        if not gts:                                  # negativa verificada
            fp += len(ps)
            continue
        if not ps:
            fn += len(gts)
            continue
        if kind == "box":
            iou = box_iou_matrix([p["bbox"] for p in ps], [g["bbox_xyxy"] for g in gts])
        else:
            if any("mask" not in p for p in ps):
                raise KeyError("kind='mask' requiere p['mask'] en todas las predicciones evaluables")
            iou = mask_iou_matrix([p["mask"] for p in ps], [g["mask_np"] for g in gts])
        taken: set[int] = set()
        for i in range(len(ps)):
            best, best_iou = -1, -1.0
            for j in range(len(gts)):
                if j in taken:
                    continue
                v = iou[i, j]
                ok = v > iou_thresh if strict else v >= iou_thresh
                if ok and v > best_iou:
                    best, best_iou = j, v
            if best >= 0:
                taken.add(best)
                tp += 1
            elif cat not in not_exh:
                fp += 1
        fn += len(gts) - len(taken)
    ignored = sum(len(v) for c, v in pred_by_cat.items() if c not in gt_by_cat and c not in neg)
    return {"tp": tp, "fp": fp, "fn": fn, "ignored_preds": ignored}


def evaluate_dataset(preds_by_image: dict[Any, list[dict]], gt_by_image: dict[Any, dict],
                     iou_thresh: float = 0.5, kind: str = "box", strict: bool = False) -> dict:
    """
    F1 micro (suma global de TP/FP/FN) y F1 macro (promedio por imagen).
    La métrica oficial de Kaggle no está especificada: reportar ambas y vigilar que no diverjan.
    """
    tp = fp = fn = ignored = 0
    per_image = {}
    for key, gt in gt_by_image.items():
        r = evaluate_image(preds_by_image.get(key, []), gt, iou_thresh, kind, strict)
        per_image[key] = {**r, **f1_from_counts(r["tp"], r["fp"], r["fn"])}
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]; ignored += r["ignored_preds"]
    micro = f1_from_counts(tp, fp, fn)
    macro = float(np.mean([v["f1"] for v in per_image.values()])) if per_image else 0.0
    return {**micro, "f1_micro": micro["f1"], "f1_macro": macro, "ignored_preds": ignored,
            "n_images": len(per_image), "per_image": per_image}
