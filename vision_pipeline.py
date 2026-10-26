"""
vision_pipeline.py  —  FASES 1-3: detección (OWLv2), segmentación (SAM2), RLE, SceneJSON y relaciones.

torch / transformers / sam2 se importan de forma PEREZOSA dentro de `Perception`, de modo que
geometría, RLE, relaciones, NMS, decodificación de OWLv2 y validadores se pueden probar sin GPU.

Flujo recomendado (permite barrer parámetros sin re-correr el modelo):
    raw  = perception.run_owl_raw(img, vocab_names)         # lo caro: 1 vez por imagen (cacheable)
    dets = decode_owl_outputs(raw["logits"], raw["boxes"], w, h, **params)   # barato, puro numpy
    masks = perception.segment(img_np, [d["bbox"] for d in dets])
    scene = build_scene(image_id, w, h, dets_con_mask_y_category_id)
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------
# Constantes del contrato (scene_schema.json) y de reglas de negocio
# ----------------------------------------------------------------------
POSITIONS = ["top_left", "top_center", "top_right",
             "middle_left", "center", "middle_right",
             "bottom_left", "bottom_center", "bottom_right"]
SIZES = ["small", "medium", "large"]
PREDICATES = ["left_of", "right_of", "above", "below", "overlaps"]   # 'inside' NO existe en el esquema

MAX_OBJECTS = 300          # schema: objects.maxItems
MAX_RELATIONS = 20000      # schema: relations.maxItems
OVERLAP_IOU = 0.10

# TODO(equipo): CONFIRMAR estos umbrales con la consigna del TPI. Vienen del notebook original.
SIZE_SMALL_MAX = 0.02
SIZE_MEDIUM_MAX = 0.08


# ======================================================================
# GEOMETRÍA (una sola fuente de verdad para centroide / posición / tamaño)
# ======================================================================
def clip_box(box, w: int, h: int) -> list[float] | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def position_of(cx: float, cy: float, w: int, h: int) -> str:
    col = 0 if cx < w / 3 else (1 if cx < 2 * w / 3 else 2)
    row = 0 if cy < h / 3 else (1 if cy < 2 * h / 3 else 2)
    return POSITIONS[row * 3 + col]


def size_of(area_ratio: float) -> str:
    return "small" if area_ratio < SIZE_SMALL_MAX else ("medium" if area_ratio < SIZE_MEDIUM_MAX else "large")


def mask_tight_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def object_attributes(bbox, w: int, h: int, mask: np.ndarray | None = None,
                      geometry_source: str = "bbox") -> dict:
    """
    centroid / area_ratio / position / size.
    geometry_source='bbox' (comportamiento original) o 'mask' (centro de masa y área reales de la máscara).
    Con máscara vacía se cae a bbox.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)
    if geometry_source == "mask" and mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean() + 0.5), float(ys.mean() + 0.5)
        area = float(mask.sum())
    area_ratio = float(min(1.0, max(1e-6, area / float(w * h))))
    return {"centroid": [float(cx), float(cy)], "area_ratio": area_ratio,
            "position": position_of(cx, cy, w, h), "size": size_of(area_ratio)}


# ======================================================================
# RELACIONES ESPACIALES
# ======================================================================
def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def relation_holds(a: dict, b: dict, predicate: str) -> bool:
    """¿'a <predicate> b'? Basado en centroides (y en IoU de cajas para 'overlaps').
    Es LA definición usada tanto al generar `relations` como al responder preguntas."""
    if predicate == "left_of":
        return a["centroid"][0] < b["centroid"][0]
    if predicate == "right_of":
        return a["centroid"][0] > b["centroid"][0]
    if predicate == "above":
        return a["centroid"][1] < b["centroid"][1]
    if predicate == "below":
        return a["centroid"][1] > b["centroid"][1]
    if predicate == "overlaps":
        return _iou(a["bbox"], b["bbox"]) > OVERLAP_IOU
    raise ValueError(f"predicado desconocido: {predicate}")


def compute_relations(objects: list[dict], include_inverse: bool = False,
                      max_relations: int = MAX_RELATIONS) -> list[dict]:
    """
    Relaciones canónicas por par no ordenado: una horizontal (left_of), una vertical (above)
    y 'overlaps' si IoU > OVERLAP_IOU. Con include_inverse=True agrega right_of/below.
    Si el total excede max_relations, se conservan los objetos de mayor confianza
    (los objetos deben venir ordenados por confianza descendente).
    """
    n = len(objects)
    if n < 2:
        return []
    per_pair = []                      # (i, j, [(pred, subj, obj), ...])
    for j in range(1, n):
        for i in range(j):
            a, b = objects[i], objects[j]
            rel = []
            ax, ay = a["centroid"]; bx, by = b["centroid"]
            if ax != bx:
                s, o = (a, b) if ax < bx else (b, a)
                rel.append(("left_of", s, o))
                if include_inverse:
                    rel.append(("right_of", o, s))
            if ay != by:
                s, o = (a, b) if ay < by else (b, a)
                rel.append(("above", s, o))
                if include_inverse:
                    rel.append(("below", o, s))
            if relation_holds(a, b, "overlaps"):
                rel.append(("overlaps", a, b))
            per_pair.append((i, j, rel))

    # Mayor n_keep tal que las relaciones entre los n_keep objetos más confiables <= max_relations
    col_total = np.zeros(n, dtype=np.int64)
    for i, j, rel in per_pair:
        col_total[j] += len(rel)
    cum = np.cumsum(col_total)
    n_keep = n
    while n_keep > 1 and cum[n_keep - 1] > max_relations:
        n_keep -= 1
    out = []
    for i, j, rel in per_pair:
        if j < n_keep:
            out.extend({"subject": s["id"], "predicate": p, "object": o["id"]} for p, s, o in rel)
    return out


# ======================================================================
# RLE (formato de SALIDA: no comprimido, orden column-major, empieza con ceros)
# ======================================================================
def mask_to_rle(mask: np.ndarray) -> dict:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"la máscara debe ser 2D (H,W), llegó {m.shape}")
    h, w = m.shape
    flat = (m > 0).astype(np.uint8).flatten(order="F")
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], change, [flat.size]))).tolist()
    if flat[0] == 1:
        counts.insert(0, 0)
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def validate_rle(rle: Any, h: int, w: int, mask: np.ndarray | None = None) -> list[str]:
    """Paso 2.2. Lista de errores (vacía = OK). Si se pasa `mask`, verifica el round-trip exacto."""
    errs: list[str] = []
    if not isinstance(rle, dict) or set(rle) < {"size", "counts"}:
        return ["RLE no es dict con claves size/counts"]
    size, counts = rle["size"], rle["counts"]
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in size)):
        errs.append(f"size inválido: {size!r}")
    elif size != [h, w]:
        errs.append(f"size {size} != [H,W]=[{h},{w}] de la imagen")
    if not isinstance(counts, list) or len(counts) < 1:
        return errs + ["counts debe ser una lista no vacía (RLE NO comprimido)"]
    if not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in counts):
        errs.append("counts debe contener enteros >= 0")
        return errs
    if sum(counts) != h * w:
        errs.append(f"sum(counts)={sum(counts)} != H*W={h * w}")
        return errs
    if mask is not None:
        if mask.shape != (h, w):
            errs.append(f"máscara {mask.shape} != ({h},{w})")
        elif not np.array_equal(rle_to_mask(rle), (mask > 0).astype(np.uint8)):
            errs.append("round-trip RLE -> máscara no coincide con la máscara original")
    return errs


# ======================================================================
# NMS y decodificación de OWLv2 (numpy puro)
# ======================================================================
def _box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Suprime cajas con IoU > iou_thr respecto de una de mayor score (igual que torchvision.ops.nms)."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = _box_iou_np(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def batched_nms_numpy(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float) -> np.ndarray:
    """NMS por categoría (equivalente a torchvision.ops.batched_nms)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    offset = labels.astype(np.float64) * (boxes.max() + 1.0)
    return nms_numpy(boxes + offset[:, None], scores, iou_thr)


def decode_owl_outputs(logits, pred_boxes, img_w: int, img_h: int, *,
                       score_thr: float = 0.2,
                       mode: str = "per_class",
                       nms_mode: str = "batched",
                       nms_iou: float = 0.5,
                       agnostic_nms_iou: float | None = None,
                       pre_nms_topk: int = 3000,
                       max_per_image: int = 100,
                       padded_square: bool = True) -> list[dict]:
    """
    Convierte la salida cruda de OWLv2 en detecciones [{bbox, score, label}].

    logits      (P, Q)  logits de clasificación por parche y consulta de texto
    pred_boxes  (P, 4)  cajas (cx, cy, w, h) normalizadas respecto de la imagen CUADRADA con padding
    mode        'per_class': cada par (parche, clase) con score >= thr es candidato  [recomendado]
                'argmax'   : una sola clase por parche (lo que hace post_process_object_detection)
    nms_mode    'batched' (por categoría) | 'global' (todas las clases juntas)
    agnostic_nms_iou: si se da (y nms_mode='batched'), segunda pasada sin categoría con ese IoU
    padded_square: OWLv2 rellena la imagen a cuadrado (abajo/derecha). Las cajas normalizadas
                   deben escalarse con S=max(w,h) en ambos ejes, NO con (w, h).
                   False reproduce el comportamiento del notebook original (target_sizes=(h,w)).
    """
    if mode not in ("per_class", "argmax") or nms_mode not in ("batched", "global"):
        raise ValueError("mode / nms_mode inválidos")
    logits = np.asarray(logits, dtype=np.float32)
    boxes = np.asarray(pred_boxes, dtype=np.float32)
    scores = 1.0 / (1.0 + np.exp(-logits))
    n_patch = scores.shape[0]

    if mode == "argmax":
        lab = scores.argmax(1)
        sc = scores[np.arange(n_patch), lab]
        keep = sc >= score_thr
        pi, qi, sc = np.flatnonzero(keep), lab[keep], sc[keep]
    else:
        pi, qi = np.nonzero(scores >= score_thr)
        sc = scores[pi, qi]
    if sc.size == 0:
        return []
    if sc.size > pre_nms_topk:
        top = np.argsort(-sc)[:pre_nms_topk]
        pi, qi, sc = pi[top], qi[top], sc[top]

    cx, cy, bw, bh = boxes[pi].T.astype(np.float64)
    sx, sy = (max(img_w, img_h),) * 2 if padded_square else (img_w, img_h)
    xyxy = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy], axis=1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)
    ok = ((xyxy[:, 2] - xyxy[:, 0]) >= 1.0) & ((xyxy[:, 3] - xyxy[:, 1]) >= 1.0)
    xyxy, sc, qi = xyxy[ok], sc[ok], qi[ok]
    if sc.size == 0:
        return []

    if nms_mode == "global":
        keep = nms_numpy(xyxy, sc, nms_iou)
    else:
        keep = batched_nms_numpy(xyxy, sc, qi, nms_iou)
        if agnostic_nms_iou is not None:
            keep = keep[nms_numpy(xyxy[keep], sc[keep], agnostic_nms_iou)]
    keep = keep[np.argsort(-sc[keep], kind="stable")][:min(max_per_image, MAX_OBJECTS)]
    return [{"bbox": [float(v) for v in xyxy[k]], "score": float(sc[k]), "label": int(qi[k])} for k in keep]


# ======================================================================
# SCENEJSON
# ======================================================================
def build_scene(image_id: str, w: int, h: int, dets: list[dict], *,
                geometry_source: str = "bbox", bbox_source: str = "owl",
                include_inverse: bool = False) -> dict:
    """
    dets: [{"bbox", "score", "category_id", "mask": ndarray(H,W)}]
    bbox_source: 'owl' (caja del detector) | 'mask' (caja ajustada a la máscara de SAM2)
    """
    dets = sorted(dets, key=lambda d: -float(d["score"]))[:MAX_OBJECTS]
    objects = []
    for d in dets:
        mask = np.asarray(d["mask"]).astype(np.uint8)
        bbox = d["bbox"]
        if bbox_source == "mask":
            bbox = mask_tight_bbox(mask) or bbox
        bbox = clip_box(bbox, w, h)
        if bbox is None:
            continue
        attrs = object_attributes(bbox, w, h, mask, geometry_source)
        objects.append({
            "id": f"obj_{len(objects)}",
            "category_id": int(d["category_id"]),
            "confidence": float(min(1.0, max(0.0, d["score"]))),
            "bbox": [float(v) for v in bbox],
            "mask": mask_to_rle(mask),
            **attrs,
        })
    return {"image": {"id": str(image_id), "width": int(w), "height": int(h)},
            "objects": objects,
            "relations": compute_relations(objects, include_inverse=include_inverse),
            "questions": [], "questions_llm": []}


def validate_scene(scene: dict, schema: dict | None = None) -> list[str]:
    """
    Valida contra el esquema OFICIAL (scene_schema.json) + chequeos de consistencia interna.
    Devuelve lista de errores (vacía = OK).
    """
    errs: list[str] = []
    if schema is not None:
        from jsonschema import Draft202012Validator
        for e in Draft202012Validator(schema).iter_errors(scene):
            errs.append(f"schema: {'/'.join(map(str, e.absolute_path))}: {e.message[:120]}")
        if errs:
            return errs[:25]

    img = scene["image"]
    w, h = img["width"], img["height"]
    ids = [o["id"] for o in scene["objects"]]
    if len(set(ids)) != len(ids):
        errs.append("ids de objeto duplicados")
    for o in scene["objects"]:
        t = o["id"]
        x1, y1, x2, y2 = o["bbox"]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            errs.append(f"{t}: bbox fuera de la imagen o degenerado {o['bbox']}")
        errs += [f"{t}: {e}" for e in validate_rle(o["mask"], h, w)]
        cx, cy = o["centroid"]
        if not (0 <= cx <= w and 0 <= cy <= h):
            errs.append(f"{t}: centroide fuera de la imagen")
        if o["position"] != position_of(cx, cy, w, h):
            errs.append(f"{t}: position no coincide con el centroide")
        if o["size"] != size_of(o["area_ratio"]):
            errs.append(f"{t}: size no coincide con area_ratio")
    idset = set(ids)
    for r in scene["relations"]:
        if r["subject"] not in idset or r["object"] not in idset:
            errs.append(f"relación con id inexistente: {r}")
    for key in ("questions", "questions_llm"):
        for q in scene.get(key, []):
            bad = [e for e in q.get("evidence", []) if e not in idset]
            if bad:
                errs.append(f"{key}/{q.get('question_id')}: evidence con ids inexistentes {bad}")
    return errs[:25]


# ======================================================================
# CACHÉ DE SALIDAS CRUDAS DE OWLv2 (para barrer thresholds sin volver a correr el modelo)
# ======================================================================
def save_raw(path: str | Path, raw: dict) -> None:
    np.savez_compressed(path, logits=raw["logits"].astype(np.float16), boxes=raw["boxes"].astype(np.float32))


def load_raw(path: str | Path) -> dict:
    z = np.load(path)
    return {"logits": z["logits"].astype(np.float32), "boxes": z["boxes"].astype(np.float32)}


# ======================================================================
# MODELOS (importaciones perezosas)
# ======================================================================
SAM2_VARIANTS = {   # (config, checkpoint) de la release 072824, la misma del notebook original
    "tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "base_plus": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/{ckpt}"


def ensure_sam2_checkpoint(variant: str = "tiny", models_dir: str = "/kaggle/working/models") -> tuple[str, str]:
    """Paso 2.1: descarga a una ruta ABSOLUTA, independiente del directorio de ejecución."""
    cfg, ckpt = SAM2_VARIANTS[variant]
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, ckpt)
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        urllib.request.urlretrieve(SAM2_URL.format(ckpt=ckpt), path)
    return cfg, path


class Perception:
    """OWLv2 + SAM2. No instanciar en tests sin GPU/pesos."""

    def __init__(self, detector: str = "google/owlv2-base-patch16-ensemble", sam2_variant: str = "tiny",
                 models_dir: str = "/kaggle/working/models", device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(detector)
        self.detector = Owlv2ForObjectDetection.from_pretrained(
            detector, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        cfg, ckpt = ensure_sam2_checkpoint(sam2_variant, models_dir)
        self.sam = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=self.device))

    def run_owl_raw(self, image_pil, vocab_names: list[str], template: str = "a {}") -> dict:
        """Lo caro: logits (P,Q) y cajas (P,4) crudos. Cacheable con save_raw/load_raw."""
        texts = [template.format(n) for n in vocab_names]
        inputs = self.processor(text=[texts], images=image_pil, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.detector(**inputs)
        return {"logits": out.logits[0].float().cpu().numpy(), "boxes": out.pred_boxes[0].float().cpu().numpy()}

    def segment(self, image_np: np.ndarray, boxes_xyxy: list, multimask: bool = False, chunk: int = 24,
                point_prompt: bool = False) -> list[np.ndarray]:
        """Una máscara bool (H,W) por caja. Con multimask=True elige la de mayor IoU predicho por SAM2.
        point_prompt=True agrega el centro de la caja como punto positivo [NO PROBADO en GPU; el centro puede
        caer fuera del objeto en aros/objetos huecos: medir antes de adoptar]."""
        if not len(boxes_xyxy):
            return []
        h, w = image_np.shape[:2]
        self.sam.set_image(image_np)
        amp = self.torch.float16 if self.device == "cuda" else self.torch.bfloat16
        masks_out: list[np.ndarray] = []
        for i in range(0, len(boxes_xyxy), chunk):
            b = np.asarray(boxes_xyxy[i:i + chunk], dtype=np.float32)
            pc = pl = None
            if point_prompt:
                pc = ((b[:, :2] + b[:, 2:]) / 2.0)[:, None, :]
                pl = np.ones((len(b), 1), dtype=np.int32)
            with self.torch.autocast(device_type=self.device, dtype=amp), self.torch.inference_mode():
                m, iou, _ = self.sam.predict(point_coords=pc, point_labels=pl, box=b, multimask_output=multimask)
            m = np.asarray(m.detach().cpu().numpy() if hasattr(m, "detach") else m).reshape(len(b), -1, h, w)
            iou = np.asarray(iou.detach().cpu().numpy() if hasattr(iou, "detach") else iou).reshape(len(b), -1)
            best = iou.argmax(1)
            masks_out.extend(m[np.arange(len(b)), best] > 0.5)
        return masks_out
