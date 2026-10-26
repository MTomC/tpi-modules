"""
pipeline_runner.py — FASE 6: ensamblaje, trazabilidad y submission.csv.

  Tracer                      registra tiempos, parámetros, conteos y errores por etapa (try/except por etapa)
  process_one(...)            = procesar_imagen_y_preguntar: imagen + preguntas -> (scene, trace)
  run_submission(...)         recorre sample_submission.csv, valida cada SceneJSON y escribe el CSV + traza

Principio: una etapa que falla NO tumba el pipeline cuando fail_fast=False; el error queda en la traza
y la escena se completa con lo que haya (nunca con tamaño 0: el esquema exige width/height >= 1).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import vision_pipeline as vp


class Tracer:
    def __init__(self, fail_fast: bool = False):
        self.fail_fast = fail_fast
        self.stage_seconds: dict[str, float] = {}
        self.errors: list[dict] = []
        self.info: dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:                                    # noqa: BLE001
            self.errors.append({"stage": name, "error": f"{type(exc).__name__}: {exc}"})
            if self.fail_fast:
                raise
        finally:
            self.stage_seconds[name] = round(time.perf_counter() - t0, 3)

    def as_dict(self) -> dict:
        return {"seconds": self.stage_seconds, "errors": self.errors, **self.info}


def group_questions(qa_rows: list[dict]) -> dict[str, list[dict]]:
    """Agrupa por id de imagen en formato 'tpi_0005' (el mismo que usa sample_submission para las filas dev)."""
    out = defaultdict(list)
    for q in qa_rows:
        out[f"tpi_{int(q['image_id']):04d}"].append(q)
    return dict(out)


def empty_scene(image_id: str, w: int, h: int) -> dict:
    return {"image": {"id": str(image_id), "width": int(w), "height": int(h)},
            "objects": [], "relations": [], "questions": [], "questions_llm": []}


def process_one(image_id: str, image_path: str | Path, perception, vocab: dict, params: dict, engine,
                questions: list[dict] | None = None, llm=None, schema: dict | None = None,
                tracer: Tracer | None = None, template: str = "a {}", sam_multimask: bool = False,
                geometry_source: str = "bbox", bbox_source: str = "owl", include_inverse: bool = False,
                raw_cache_dir: str | Path | None = None) -> tuple[dict, dict]:
    from PIL import Image
    tr = tracer or Tracer()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)
    tr.info.update({"image_id": image_id, "size": [w, h], "params": dict(params), "template": template,
                    "geometry_source": geometry_source, "bbox_source": bbox_source})
    raw, dets, masks, scene = None, [], [], None

    with tr.stage("owl"):
        cache = Path(raw_cache_dir) / f"{image_id}.npz" if raw_cache_dir else None
        if cache is not None and cache.exists():
            raw = vp.load_raw(cache)
        else:
            raw = perception.run_owl_raw(img, vocab["vocab_names"], template)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                vp.save_raw(cache, raw)
    with tr.stage("decode"):
        if raw is not None:
            dets = vp.decode_owl_outputs(raw["logits"], raw["boxes"], w, h, **params)
        tr.info["n_detections"] = len(dets)
    with tr.stage("sam2"):
        masks = perception.segment(img_np, [d["bbox"] for d in dets], multimask=sam_multimask) if dets else []
        tr.info["n_masks"] = len(masks)
        tr.info["n_empty_masks"] = int(sum(not np.asarray(m).any() for m in masks))
    with tr.stage("scene"):
        full = [{"category_id": int(vocab["vocab_ids"][d["label"]]), "bbox": d["bbox"], "score": d["score"], "mask": m}
                for d, m in zip(dets, masks)]
        scene = vp.build_scene(image_id, w, h, full, geometry_source=geometry_source, bbox_source=bbox_source,
                               include_inverse=include_inverse)
        tr.info["n_objects"] = len(scene["objects"])
        tr.info["n_relations"] = len(scene["relations"])
    if scene is None:
        scene = empty_scene(image_id, w, h)

    qs = questions or []
    with tr.stage("questions"):
        scene["questions"] = [engine.answer(scene, q) for q in qs]
        tr.info["n_questions"] = len(qs)
    if llm is not None and qs:
        with tr.stage("questions_llm"):
            scene["questions_llm"] = [llm.answer(scene, q) for q in qs]
            tr.info["llm_stats"] = dict(llm.stats)
    with tr.stage("validate"):
        errs = vp.validate_scene(scene, schema)
        tr.info["validation_errors"] = errs
    return scene, tr.as_dict()


def _find_image(image_id: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = Path(d) / f"{image_id}{ext}"
            if p.exists():
                return p
    return None


def run_submission(sample_df, image_dirs: list[str | Path], perception, vocab: dict, params: dict, engine,
                   qa_rows: list[dict], llm=None, schema: dict | None = None, out_csv: str | Path = "submission.csv",
                   trace_json: str | Path | None = None, fail_fast: bool = False, **process_kwargs):
    """Procesa las 80 filas, valida y escribe el CSV con el formato exacto (image_id, SceneJSON)."""
    import pandas as pd
    qa_by_key = group_questions(qa_rows)
    dirs = [Path(d) for d in image_dirs]
    rows, traces = [], {}
    for n, image_id in enumerate(sample_df["image_id"].astype(str), 1):
        tr = Tracer(fail_fast=fail_fast)
        path = _find_image(image_id, dirs)
        if path is None:
            raise FileNotFoundError(f"No se encontró la imagen {image_id} en {dirs}")
        scene, trace = process_one(image_id, path, perception, vocab, params, engine, qa_by_key.get(image_id),
                                   llm, schema, tr, **process_kwargs)
        rows.append({"image_id": image_id, "SceneJSON": json.dumps(scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False)})
        traces[image_id] = trace
        flag = "ERR" if trace["errors"] else ("val!" if trace.get("validation_errors") else "ok")
        print(f"[{n}/{len(sample_df)}] {image_id}: {flag} | obj={trace.get('n_objects')} | {sum(trace['seconds'].values()):.1f}s")

    sub = pd.DataFrame(rows)
    # --- chequeos finales del contrato ---
    assert list(sub.columns) == list(sample_df.columns) == ["image_id", "SceneJSON"], "columnas distintas al sample"
    assert len(sub) == len(sample_df) and list(sub["image_id"]) == list(sample_df["image_id"].astype(str)), "ids/orden"
    for s in sub["SceneJSON"]:
        json.loads(s)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    summary = {"n_images": len(sub),
               "images_with_stage_errors": [k for k, t in traces.items() if t["errors"]],
               "images_with_validation_errors": [k for k, t in traces.items() if t.get("validation_errors")],
               "total_seconds": round(sum(sum(t["seconds"].values()) for t in traces.values()), 1)}
    if trace_json:
        Path(trace_json).write_text(json.dumps({"summary": summary, "traces": traces}, ensure_ascii=False, indent=1, default=str))
    print("Resumen:", summary)
    return sub, traces, summary
