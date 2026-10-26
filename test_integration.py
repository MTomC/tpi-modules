"""
test_integration.py — humo end-to-end con un detector/segmentador FALSOS (sin GPU, sin pesos, sin dataset).
Verifica el cableado entre módulos: decode -> sweep -> máscaras -> escena -> motor -> LLM(mock) -> CSV.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dataset_utils as du
import vision_pipeline as vp
import query_engine as qe
import llm_wrapper as lw
import experiments as ex
import pipeline_runner as pr

SCHEMA = json.loads((Path(__file__).parent / "scene_schema.json").read_text())
VOCAB = {"vocab_ids": [1, 2, 3], "vocab_names": ["person", "chair", "dog"],
         "name_to_id": {"person": 1, "chair": 2, "dog": 3}, "id_to_name": {1: "person", 2: "chair", 3: "dog"}}

# 3 imágenes apaisadas 160x120 con GT en rectángulos
IMGS = {
    1: [(1, [10, 40, 50, 110]), (2, [70, 50, 110, 100]), (3, [120, 70, 155, 110])],
    2: [(1, [20, 10, 60, 60]), (2, [90, 60, 150, 115])],
    3: [(3, [5, 5, 40, 30]), (3, [100, 80, 150, 118]), (2, [60, 40, 90, 70])],
}
NEG = {1: [], 2: [3], 3: [1]}          # imagen 2 sin perro; imagen 3 sin persona (negativas verificadas)


def make_dev():
    images, anns, aid = [], [], 1
    for i, items in IMGS.items():
        images.append({"id": i, "file_name": f"tpi_{i:04d}.jpg", "width": 160, "height": 120,
                       "neg_category_ids": NEG[i], "not_exhaustive_category_ids": []})
        for c, (x1, y1, x2, y2) in items:
            anns.append({"id": aid, "image_id": i, "category_id": c, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                         "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]})
            aid += 1
    cats = [{"id": 1, "name": "person"}, {"id": 2, "name": "chair"}, {"id": 3, "name": "dog"}]
    return du.DevData({"images": images, "annotations": anns, "categories": cats})


def fake_raw(items, w, h, extra_false_positive=True):
    """Salida 'cruda' coherente con OWLv2: cajas normalizadas a la imagen CUADRADA con padding (S=max(w,h))."""
    S = max(w, h)
    n = 40
    logits = np.full((n, 3), -10.0, np.float32)
    boxes = np.tile(np.array([[.5, .5, .05, .05]], np.float32), (n, 1))
    for k, (c, (x1, y1, x2, y2)) in enumerate(items):
        boxes[k] = [(x1 + x2) / 2 / S, (y1 + y2) / 2 / S, (x2 - x1) / S, (y2 - y1) / S]
        logits[k, c - 1] = np.log(.8 / .2)
    if extra_false_positive:                       # un FP de baja confianza en una categoría evaluable
        boxes[30] = [.9, .2, .1, .1]
        logits[30, 0] = np.log(.12 / .88)
    return {"logits": logits, "boxes": boxes}


def fake_segment_factory(dev):
    def seg(image_np, boxes):
        h, w = image_np.shape[:2]
        out = []
        for b in boxes:
            m = np.zeros((h, w), np.uint8)
            m[int(round(b[1])):int(round(b[3])), int(round(b[0])):int(round(b[2]))] = 1
            out.append(m)
        return out
    return seg


def test_sweep_detecta_el_bug_de_padding_y_el_mejor_threshold():
    dev = make_dev()
    raw = {i: fake_raw(IMGS[i], 160, 120) for i in IMGS}
    grid = ex.make_grid(score_thr=[0.1, 0.5], padded_square=[True, False], mode=["per_class"], nms_mode=["batched"])
    df = ex.sweep_detection(raw, dev, VOCAB["vocab_ids"], grid)
    best = df.iloc[0]
    assert best["padded_square"] == True and best["score_thr"] == 0.5 and best["F1_micro"] == 1.0, df
    worst_pad = df[(df.padded_square == False)]["F1_micro"].max()
    assert worst_pad < 0.6, df           # sin corregir el padding, las cajas caen bajo IoU 0.5 en imágenes apaisadas
    # el FP de baja confianza (persona 0.12) solo duele a threshold 0.1 y solo donde 'person' es evaluable
    low = df[(df.padded_square == True) & (df.score_thr == 0.1)].iloc[0]
    assert low["FP"] >= 1 and low["F1_micro"] < 1.0


def test_mascaras_y_qa_proxy():
    dev = make_dev()
    raw = {i: fake_raw(IMGS[i], 160, 120, extra_false_positive=False) for i in IMGS}
    params = dict(score_thr=0.5, mode="per_class", nms_mode="batched", padded_square=True)
    loader = lambda i: np.zeros((120, 160, 3), np.uint8)
    r = ex.evaluate_with_masks(raw, dev, VOCAB["vocab_ids"], params, loader, fake_segment_factory(dev))
    assert r["box_f1"] == 1.0 and r["mask_f1"] == 1.0, r["box"]
    eng = qe.QueryEngine(VOCAB["name_to_id"])
    qa = [{"question_id": "A", "type": "count", "image_id": 3, "question": "?", "class_en": ["dog"]},
          {"question_id": "B", "type": "existence", "image_id": 2, "question": "?", "class_en": ["dog"]},
          {"question_id": "C", "type": "position", "image_id": 1, "question": "?", "class_en": ["chair"]},
          {"question_id": "D", "type": "relation", "image_id": 1, "question": "¿... a la izquierda de ...?", "class_en": ["person", "dog"]}]
    expected = ex.expected_answers(dev, qa, eng)
    assert expected["A"] == 2 and expected["B"] == "no" and expected["D"] == "yes"
    scenes = {i: ex.scene_from_gt(dev, i) for i in IMGS}      # escena perfecta -> 100%
    df, acc, by_type = ex.qa_accuracy(scenes, qa, eng, expected)
    assert acc == 1.0
    audit = du.audit_qa_questions(dev, qa)
    assert set(audit["status"]) <= {"positive", "negative", "unverified"} and (audit[audit.question_id == "B"].status == "negative").all()


class FakePerception:
    def run_owl_raw(self, img, names, template="a {}"):
        w, h = img.size
        return fake_raw(IMGS[int(Path(img.filename).stem.split("_")[1])] if getattr(img, "filename", None) else IMGS[1], w, h, False)

    def segment(self, image_np, boxes, multimask=False):
        return fake_segment_factory(None)(image_np, boxes)


def test_submission_end_to_end():
    import pandas as pd
    from PIL import Image
    dev = make_dev()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dev").mkdir(); (td / "test").mkdir()
        for i in IMGS:
            Image.new("RGB", (160, 120), (90, 90, 90)).save(td / "dev" / f"tpi_{i:04d}.jpg")
        Image.new("RGB", (160, 120), (50, 50, 50)).save(td / "test" / "test_0001.jpg")
        sample = pd.DataFrame({"image_id": ["test_0001", "tpi_0001", "tpi_0003"], "SceneJSON": ["{}"] * 3})
        qa = [{"question_id": "DEV-1-Q1", "type": "count", "image_id": 1, "question": "¿Cuántas instancias de \"silla\"?", "class_en": ["chair"]},
              {"question_id": "DEV-1-Q2", "type": "relation", "image_id": 1, "question": "¿La persona está a la izquierda del perro?", "class_en": ["person", "dog"]},
              {"question_id": "DEV-1-Q3", "type": "description", "image_id": 1, "question": "Describí la imagen", "class_en": []}]
        eng = qe.QueryEngine(VOCAB["name_to_id"])
        llm = lw.LLMWrapper(lambda p: "{}", eng, VOCAB["vocab_names"])          # LLM 'caído' -> reglas + plantillas
        # el stub necesita saber qué imagen es: parcheamos run_owl_raw para usar el nombre del archivo
        class P(FakePerception):
            def run_owl_raw(self, img, names, template="a {}"):
                return fake_raw(IMGS[1], *img.size, False)
        out_csv = td / "out" / "submission.csv"
        sub, traces, summary = pr.run_submission(
            sample, [td / "test", td / "dev"], P(), VOCAB, dict(score_thr=0.5, mode="per_class", nms_mode="batched"),
            eng, qa, llm=llm, schema=SCHEMA, out_csv=out_csv, trace_json=td / "trace.json")
        assert summary["images_with_stage_errors"] == [] and summary["images_with_validation_errors"] == []
        back = pd.read_csv(out_csv)
        assert list(back.columns) == ["image_id", "SceneJSON"] and len(back) == 3
        scene = json.loads(back.loc[back.image_id == "tpi_0001", "SceneJSON"].iloc[0])
        assert len(scene["questions"]) == 3 and len(scene["questions_llm"]) == 3
        assert scene["questions"][0]["normalized_answer"] == 1 and scene["questions_llm"][1]["normalized_answer"] == "yes"
        assert "interpreted_query" in scene["questions_llm"][0]
        assert json.loads(back.loc[back.image_id == "test_0001", "SceneJSON"].iloc[0])["questions"] == []
        assert vp.validate_scene(scene, SCHEMA) == []


def test_fallo_de_etapa_no_tumba_y_conserva_tamano_real():
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jpg"
        Image.new("RGB", (200, 100)).save(p)

        class Boom:
            def run_owl_raw(self, *a, **k): raise RuntimeError("OOM simulado")
            def segment(self, *a, **k): return []
        eng = qe.QueryEngine(VOCAB["name_to_id"])
        q = [{"question_id": "q", "type": "count", "question": "?", "class_en": ["dog"]}]
        scene, trace = pr.process_one("x", p, Boom(), VOCAB, dict(score_thr=.5), eng, q, schema=SCHEMA, tracer=pr.Tracer(fail_fast=False))
        assert trace["errors"][0]["stage"] == "owl" and scene["image"]["width"] == 200 and scene["image"]["height"] == 100
        assert scene["questions"][0]["normalized_answer"] == 0 and trace["validation_errors"] == []
        try:
            pr.process_one("x", p, Boom(), VOCAB, dict(score_thr=.5), eng, q, tracer=pr.Tracer(fail_fast=True))
            raise AssertionError("fail_fast debía propagar")
        except RuntimeError:
            pass


if __name__ == "__main__":
    for fn in (test_sweep_detecta_el_bug_de_padding_y_el_mejor_threshold, test_mascaras_y_qa_proxy,
               test_submission_end_to_end, test_fallo_de_etapa_no_tumba_y_conserva_tamano_real):
        fn()
        print("  OK ", fn.__name__)
    print("\nintegración OK")
