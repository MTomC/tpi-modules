# =====================================================================
# notebook_driver.py — celdas (# %%) en el ORDEN del plan por fases.
# Cada celda es independiente de las anteriores salvo por las variables que declara.
# Las celdas marcadas [GPU] NO pudieron probarse fuera de Kaggle: correrlas con 2-3 imágenes primero.
#
# Cómo cargar los módulos en Kaggle (elegir una):
#   a) subir los .py como Dataset ("tpi-modules") y:  sys.path.append("/kaggle/input/tpi-modules")
#   b) %%writefile dataset_utils.py (y los demás) en celdas previas.
# =====================================================================

# %% [Setup]
import json, sys, os
from pathlib import Path
import numpy as np
sys.path.append("/kaggle/input/tpi-modules")          # <- ajustar
import dataset_utils as du
import vision_pipeline as vp
import query_engine as qe
import llm_wrapper as lw
import experiments as ex
import pipeline_runner as pr

paths = du.Paths.from_base()
dev = du.DevData.load(paths.annotations)
vocab = du.load_categories(paths.categories)
qa_rows = du.read_jsonl(paths.qa_file)
schema = json.loads(paths.schema.read_text())
engine = qe.QueryEngine(vocab["name_to_id"])
OUT = Path("/kaggle/working"); (OUT / "cache_raw").mkdir(exist_ok=True)
print(len(dev.images), "imágenes dev |", len(vocab["vocab_ids"]), "categorías |", len(qa_rows), "preguntas")

# %% [FASE 0.1] EDA (sin torch). Mirar sobre todo: neg_ids/imagen, % anotaciones en categorías no exhaustivas,
#               proporción small/medium/large con los umbrales actuales, y si iscrowd > 0.
rep = du.eda_report(dev, vp.SIZE_SMALL_MAX, vp.SIZE_MEDIUM_MAX)
for k, v in rep.items():
    if k not in ("images_df", "annotations_df"):
        print(f"\n--- {k} ---\n{v}")
du.plot_gt_samples(dev, paths.dev_images, n=5, draw_masks=True, save_dir=OUT / "eda")
audit = du.audit_qa_questions(dev, qa_rows)          # ¿las preguntas apuntan a instancias únicas?
print(audit.groupby(["type", "status"]).n_gt_instances.agg(["count", "min", "max"]))

# %% [FASE 0.2] Pruebas del evaluador y del resto de módulos (CPU, segundos)
import subprocess
print(subprocess.run([sys.executable, "/kaggle/input/tpi-modules/test_all.py"], capture_output=True, text=True).stdout[-600:])

# %% [FASE 1.2] [GPU] Correr OWLv2 UNA vez sobre las 120 imágenes dev y cachear lo crudo
from PIL import Image
perception = vp.Perception(sam2_variant="tiny", models_dir="/kaggle/working/models")
TEMPLATE = "a {}"            # probar también "a photo of a {}" (repetir el caché con otro nombre de carpeta)
raw_by_img = {}
for i, info in dev.images.items():
    f = OUT / "cache_raw" / f"{dev.image_key(i)}.npz"
    if f.exists():
        raw_by_img[i] = vp.load_raw(f)
    else:
        raw_by_img[i] = perception.run_owl_raw(Image.open(paths.dev_images / info["file_name"]).convert("RGB"),
                                               vocab["vocab_names"], TEMPLATE)
        vp.save_raw(f, raw_by_img[i])

# %% [FASE 1.2] Barrido OFFLINE de parámetros (segundos). Hipótesis a contrastar, en este orden:
#   1) padded_square True vs False  (¿el escalado de cajas de OWLv2 con padding es la causa del F1 bajo?)
#   2) mode argmax vs per_class     3) nms_mode global vs batched   4) score_thr   5) nms_iou
grid = ex.make_grid(padded_square=[True, False], mode=["argmax", "per_class"], nms_mode=["global", "batched"],
                    score_thr=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30], nms_iou=[0.5], max_per_image=[100])
df_det = ex.sweep_detection(raw_by_img, dev, vocab["vocab_ids"], grid)
df_det.to_csv(OUT / "sweep_detection.csv", index=False)
print(df_det.head(15).to_string())
# Refinamiento fino alrededor del ganador (nms_iou, max_per_image, agnostic_nms_iou):
best = df_det.iloc[0].to_dict()
grid2 = ex.make_grid(padded_square=[best["padded_square"]], mode=[best["mode"]], nms_mode=["batched", "global"],
                     score_thr=[best["score_thr"] - .05, best["score_thr"], best["score_thr"] + .05],
                     nms_iou=[0.4, 0.5, 0.6, 0.7], max_per_image=[50, 100, 150])
print(ex.sweep_detection(raw_by_img, dev, vocab["vocab_ids"], grid2).head(10).to_string())

# %% [FASE 1.2 — decisión] Congelar los parámetros ganadores como constante global
PARAMS = dict(score_thr=0.20, mode="per_class", nms_mode="batched", nms_iou=0.5, max_per_image=100, padded_square=True)  # <- reemplazar
# Visual de control (GT verde / pred roja) en 5 imágenes:
sample_ids = sorted(dev.images)[:5]
preds_vis = {i: ex.dets_to_preds(vp.decode_owl_outputs(raw_by_img[i]["logits"], raw_by_img[i]["boxes"],
                                                       int(dev.images[i]["width"]), int(dev.images[i]["height"]), **PARAMS),
                                 vocab["vocab_ids"]) for i in sample_ids}
ex.plot_gt_vs_pred(dev, paths.dev_images, preds_vis, sample_ids)

# %% [FASE 2.3] [GPU] Máscaras: F1 de cajas y de máscaras con los PARAMS elegidos
loader = lambda i: np.array(Image.open(paths.dev_images / dev.images[i]["file_name"]).convert("RGB"))
res = {}
for src in ("owl", "mask"):             # ¿la caja ajustada a la máscara de SAM2 se parece más al GT?
    res[src] = ex.evaluate_with_masks(raw_by_img, dev, vocab["vocab_ids"], PARAMS, loader,
                                      lambda im, b: perception.segment(im, b, multimask=False), bbox_source=src)
    print(f"bbox_source={src}:  box F1={res[src]['box_f1']:.4f}  mask F1={res[src]['mask_f1']:.4f}")
# Si la máscara falla: probar multimask=True, y variantes 'small' / 'base_plus' / 'large' de SAM2 (sin tocar OWLv2).
ex.plot_gt_vs_pred(dev, paths.dev_images, res["owl"]["preds_by_img"], sample_ids, show_masks=True)

# %% [FASE 4] Motor determinista: exactitud contra el proxy GT (40 preguntas: ruido alto, no sobre-ajustar)
expected = ex.expected_answers(dev, qa_rows, engine)
qa_imgs = sorted({int(q["image_id"]) for q in qa_rows})
scenes = {}
for i in qa_imgs:
    info = dev.images[i]; w, h = int(info["width"]), int(info["height"])
    dets = vp.decode_owl_outputs(raw_by_img[i]["logits"], raw_by_img[i]["boxes"], w, h, **PARAMS)
    masks = perception.segment(loader(i), [d["bbox"] for d in dets])
    scenes[i] = vp.build_scene(dev.image_key(i), w, h,
                               [{"category_id": vocab["vocab_ids"][d["label"]], "bbox": d["bbox"], "score": d["score"], "mask": m}
                                for d, m in zip(dets, masks)])
df_qa, acc, by_type = ex.qa_accuracy(scenes, qa_rows, engine, expected)
print("QA proxy:", round(acc, 3)); print(by_type); print(df_qa[~df_qa.ok])

# %% [FASE 5] LLM. `llm_fn` es cualquier callable str -> str. SIN LLM disponible, el wrapper cae a reglas+plantillas
#            y IGUAL llena questions_llm (si queda vacío, se pierde la mitad del 10 % de QA).
def llm_fn(prompt: str) -> str:
    raise RuntimeError("conectar un LLM aquí")          # <- reemplazar (modelo local de HF o API)
llm = lw.LLMWrapper(llm_fn, engine, vocab["vocab_names"])

# %% [FASE 6] [GPU] Submission final: 60 test + 20 dev, con traza y validación contra el esquema oficial
import pandas as pd
sample = pd.read_csv(paths.sample_submission)
sub, traces, summary = pr.run_submission(
    sample, [paths.test_images, paths.dev_images], perception, vocab, PARAMS, engine, qa_rows, llm=llm,
    schema=schema, out_csv=OUT / "submission.csv", trace_json=OUT / "trace.json", fail_fast=False,
    template=TEMPLATE, raw_cache_dir=OUT / "cache_raw_submit")
