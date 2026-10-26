"""
test_all.py — pruebas aisladas de cada fase, sin GPU ni dataset real.
Ejecutar:  python test_all.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dataset_utils as du
import vision_pipeline as vp
import query_engine as qe
import llm_wrapper as lw

PASSED = []


def test(fn):
    fn()
    PASSED.append(fn.__name__)
    print(f"  OK  {fn.__name__}")
    return fn


def gt_of(anns, neg=(), not_exh=(), w=100, h=100):
    """anns: [(cat, [x1,y1,x2,y2])]"""
    info = {"id": 1, "width": w, "height": h, "file_name": "x.jpg",
            "neg_category_ids": list(neg), "not_exhaustive_category_ids": list(not_exh)}
    prepared = [{"category_id": c, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]} for c, b in anns]
    return du.prepare_gt(info, prepared)


def P(cat, box, score=0.9, mask=None):
    d = {"category_id": cat, "bbox": box, "score": score}
    if mask is not None:
        d["mask"] = mask
    return d


# ======================================================================
print("\n== FASE 0: evaluador F1 (LVIS) ==")


@test
def f1_perfecto():
    r = du.evaluate_image([P(1, [10, 10, 30, 30])], gt_of([(1, [10, 10, 30, 30])]))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert du.f1_from_counts(**{k: r[k] for k in ("tp", "fp", "fn")})["f1"] == 1.0


@test
def f1_pesimo():
    r = du.evaluate_image([P(1, [60, 60, 80, 80])], gt_of([(1, [10, 10, 30, 30])]))
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert du.f1_from_counts(0, 1, 1)["f1"] == 0.0


@test
def categoria_no_verificada_se_ignora():
    r = du.evaluate_image([P(2, [60, 60, 80, 80])], gt_of([(1, [10, 10, 30, 30])]))
    assert (r["fp"], r["fn"], r["ignored_preds"]) == (0, 1, 1)


@test
def categoria_negativa_penaliza_fp():
    r = du.evaluate_image([P(2, [60, 60, 80, 80]), P(2, [1, 1, 5, 5])], gt_of([(1, [10, 10, 30, 30])], neg=[2]))
    assert r["fp"] == 2 and r["ignored_preds"] == 0


@test
def no_exhaustiva_ignora_extras_pero_exhaustiva_penaliza():
    preds = [P(1, [10, 10, 30, 30], .9), P(1, [60, 60, 80, 80], .8)]
    g = [(1, [10, 10, 30, 30])]
    assert du.evaluate_image(preds, gt_of(g, not_exh=[1]))["fp"] == 0
    assert du.evaluate_image(preds, gt_of(g))["fp"] == 1


@test
def duplicados_cuentan_un_tp_y_un_fp():
    preds = [P(1, [10, 10, 30, 30], .9), P(1, [10, 10, 30, 30], .8)]
    r = du.evaluate_image(preds, gt_of([(1, [10, 10, 30, 30])]))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)


@test
def umbral_iou_exacto_y_strict():
    # GT 10x10 en (0,0); pred 10x10 desplazada 10/3 -> IoU exacto != 0.5, usamos caja construida a IoU=0.5:
    # pred cubre la mitad izquierda del GT: inter=50, union=100 -> IoU = 0.5 exacto
    g = gt_of([(1, [0, 0, 10, 10])])
    p = [P(1, [0, 0, 5, 10])]
    assert du.evaluate_image(p, g, 0.5, strict=False)["tp"] == 1      # >=
    assert du.evaluate_image(p, g, 0.5, strict=True)["tp"] == 0       # >
    p2 = [P(1, [0, 0, 4.9, 10])]
    assert du.evaluate_image(p2, g, 0.5)["tp"] == 0                    # IoU=0.49


@test
def valor_esperado_a_mano():
    # GT: 3 sillas(1) + 1 perro(2). Preds: 2 sillas OK, 1 silla mala, perro bien, gato(3) sin verificar.
    g = gt_of([(1, [0, 0, 10, 10]), (1, [20, 0, 30, 10]), (1, [40, 0, 50, 10]), (2, [0, 50, 20, 70])])
    preds = [P(1, [0, 0, 10, 10]), P(1, [20, 0, 30, 10]), P(1, [70, 70, 90, 90]),
             P(2, [0, 50, 20, 70]), P(3, [0, 0, 99, 99])]
    r = du.evaluate_image(preds, g)
    # TP=3 (2 sillas + perro), FP=1 (silla mala), FN=1 (3ª silla), gato ignorado
    assert (r["tp"], r["fp"], r["fn"], r["ignored_preds"]) == (3, 1, 1, 1)
    f = du.f1_from_counts(r["tp"], r["fp"], r["fn"])
    assert abs(f["precision"] - 0.75) < 1e-9 and abs(f["recall"] - 0.75) < 1e-9 and abs(f["f1"] - 0.75) < 1e-9


@test
def categoria_equivocada_misma_caja():
    # predice 'gato' (2, no verificada) sobre la silla (1): la silla queda FN y el gato se ignora
    r = du.evaluate_image([P(2, [0, 0, 10, 10])], gt_of([(1, [0, 0, 10, 10])]))
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)


@test
def mascaras_perfecta_media_y_vacia():
    h = w = 20
    m = np.zeros((h, w), np.uint8); m[0:10, 0:10] = 1
    gtm = {"category_id": 1, "bbox_xyxy": [0, 0, 10, 10], "mask_np": m}
    g = {"anns": [gtm], "neg": set(), "not_exh": set(), "image": {}}
    assert du.evaluate_image([P(1, [0, 0, 10, 10], mask=m.copy())], g, kind="mask")["tp"] == 1
    half = np.zeros_like(m); half[0:5, 0:10] = 1                       # IoU = 0.5
    assert du.evaluate_image([P(1, [0, 0, 10, 10], mask=half)], g, kind="mask")["tp"] == 1
    less = np.zeros_like(m); less[0:4, 0:10] = 1                       # IoU = 0.4
    assert du.evaluate_image([P(1, [0, 0, 10, 10], mask=less)], g, kind="mask")["tp"] == 0
    assert du.evaluate_image([P(1, [0, 0, 10, 10], mask=np.zeros_like(m))], g, kind="mask")["tp"] == 0


@test
def micro_vs_macro():
    g1 = gt_of([(1, [0, 0, 10, 10])]); g2 = gt_of([(1, [0, 0, 10, 10]), (1, [20, 20, 30, 30])])
    res = du.evaluate_dataset({"a": [P(1, [0, 0, 10, 10])], "b": []}, {"a": g1, "b": g2})
    assert (res["tp"], res["fp"], res["fn"]) == (1, 0, 2)
    assert abs(res["f1_micro"] - 0.5) < 1e-9 and abs(res["f1_macro"] - 0.5) < 1e-9   # img a=1.0, img b=0.0


# ======================================================================
print("\n== FASE 2: RLE ==")


@test
def rle_roundtrip_y_pycocotools():
    from pycocotools import mask as mu
    rng = np.random.default_rng(0)
    for shape in [(7, 5), (480, 640), (640, 425)]:
        for density in (0.0, 0.3, 1.0):
            m = (rng.random(shape) < density).astype(np.uint8)
            if density == 0.3:
                m[:, 0] = 1                                  # primer píxel en 1 (caso borde)
            rle = vp.mask_to_rle(m)
            assert vp.validate_rle(rle, *shape, mask=m) == [], (shape, density)
            dec = mu.decode(mu.frPyObjects({"size": list(shape), "counts": rle["counts"]}, *shape))
            assert np.array_equal(dec, m)                    # coincide con el decodificador oficial COCO


@test
def validador_rle_detecta_errores():
    h, w = 4, 5
    ok = {"size": [h, w], "counts": [0, 20]}
    assert vp.validate_rle(ok, h, w) == []
    assert any("size" in e for e in vp.validate_rle({"size": [w, h], "counts": [0, 20]}, h, w))      # H,W invertidos
    assert any("sum" in e for e in vp.validate_rle({"size": [h, w], "counts": [0, 19]}, h, w))
    assert any("comprimido" in e or "lista" in e for e in vp.validate_rle({"size": [h, w], "counts": "abc"}, h, w))
    assert any("enteros" in e for e in vp.validate_rle({"size": [h, w], "counts": [0, -1, 21]}, h, w))
    assert any("enteros" in e for e in vp.validate_rle({"size": [h, w], "counts": [0.0, 20.0]}, h, w))
    m = np.zeros((h, w), np.uint8); m[0, 0] = 1
    assert any("round-trip" in e for e in vp.validate_rle(ok, h, w, mask=m))


# ======================================================================
print("\n== FASE 1: decodificación OWLv2 y NMS ==")


def _fake_owl(boxes_cxcywh, class_scores, n_q=4, n_patch=50):
    """logits sintéticos: casi todo muy negativo; parches fijos con el score pedido."""
    logits = np.full((n_patch, n_q), -10.0, np.float32)
    boxes = np.tile(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), (n_patch, 1))
    for i, (b, (q, s)) in enumerate(zip(boxes_cxcywh, class_scores)):
        boxes[i] = b
        logits[i, q] = np.log(s / (1 - s))
    return logits, boxes


@test
def padding_cuadrado_vs_original():
    # imagen apaisada 640x480: S=640. Caja en y∈[0.5,0.75] normalizada al cuadrado => y px = 320..480
    logits, boxes = _fake_owl([[0.5, 0.625, 0.4, 0.25]], [(0, 0.9)])
    ok = vp.decode_owl_outputs(logits, boxes, 640, 480, score_thr=0.5, padded_square=True)[0]["bbox"]
    old = vp.decode_owl_outputs(logits, boxes, 640, 480, score_thr=0.5, padded_square=False)[0]["bbox"]
    assert np.allclose(ok, [192, 320, 448, 480])
    assert np.allclose(old, [192, 240, 448, 360])        # comportamiento previo: desplazada y achatada 25%
    gt = du.box_iou_matrix([ok], [old])[0, 0]
    assert gt < 0.5                                      # con ese sesgo, el IoU cae bajo el umbral


@test
def nms_por_categoria_vs_global():
    # mismas cajas casi idénticas con 2 categorías distintas
    logits, boxes = _fake_owl([[0.5, 0.5, 0.3, 0.3], [0.5, 0.5, 0.3, 0.3]], [(0, 0.9), (1, 0.8)])
    g = vp.decode_owl_outputs(logits, boxes, 100, 100, score_thr=0.5, nms_mode="global", mode="per_class")
    b = vp.decode_owl_outputs(logits, boxes, 100, 100, score_thr=0.5, nms_mode="batched", mode="per_class")
    assert len(g) == 1 and g[0]["label"] == 0
    assert sorted(d["label"] for d in b) == [0, 1]       # NMS por categoría conserva ambas etiquetas
    ag = vp.decode_owl_outputs(logits, boxes, 100, 100, score_thr=0.5, nms_mode="batched", agnostic_nms_iou=0.9)
    assert len(ag) == 1


@test
def argmax_pierde_la_segunda_etiqueta():
    logits = np.full((3, 2), -10.0, np.float32)
    logits[0] = [np.log(.9 / .1), np.log(.8 / .2)]       # el parche 0 tiene dos etiquetas plausibles
    boxes = np.array([[.5, .5, .2, .2]] * 3, np.float32)
    a = vp.decode_owl_outputs(logits, boxes, 100, 100, score_thr=0.5, mode="argmax", nms_mode="batched")
    p = vp.decode_owl_outputs(logits, boxes, 100, 100, score_thr=0.5, mode="per_class", nms_mode="batched")
    assert len(a) == 1 and len(p) == 2


@test
def nms_numpy_valores_a_mano():
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    s = np.array([0.9, 0.8, 0.7])
    keep = vp.nms_numpy(b, s, 0.5)               # IoU(0,1)=81/119=0.68 -> suprime la 1
    assert keep.tolist() == [0, 2]
    assert vp.nms_numpy(b, s, 0.7).tolist() == [0, 1, 2]


@test
def tope_de_objetos():
    logits = np.full((500, 3), 5.0, np.float32)
    rng = np.random.default_rng(1)
    boxes = np.stack([rng.uniform(.2, .8, 500), rng.uniform(.2, .8, 500), np.full(500, .02), np.full(500, .02)], 1).astype(np.float32)
    d = vp.decode_owl_outputs(logits, boxes, 640, 480, score_thr=0.5, max_per_image=1000, nms_iou=0.99)
    assert len(d) <= vp.MAX_OBJECTS


# ======================================================================
print("\n== FASE 3: geometría, relaciones y SceneJSON ==")


def _mk_det(cat, box, score, w, h):
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = (int(v) for v in box)
    m[y1:y2, x1:x2] = 1
    return {"category_id": cat, "bbox": [float(v) for v in box], "score": score, "mask": m}


@test
def posicion_y_tamano():
    assert vp.position_of(10, 10, 300, 300) == "top_left"
    assert vp.position_of(150, 150, 300, 300) == "center"
    assert vp.position_of(299, 299, 300, 300) == "bottom_right"
    assert vp.size_of(0.01) == "small" and vp.size_of(0.05) == "medium" and vp.size_of(0.5) == "large"


@test
def relaciones_persona_silla_perro():
    W = H = 300
    dets = [_mk_det(1, [10, 100, 60, 250], .9, W, H),      # persona (izquierda)
            _mk_det(2, [120, 120, 190, 250], .8, W, H),    # silla (centro)
            _mk_det(3, [210, 200, 290, 280], .7, W, H)]    # perro (derecha-abajo)
    scene = vp.build_scene("img", W, H, dets)
    ids = {o["category_id"]: o["id"] for o in scene["objects"]}
    rel = {(r["subject"], r["predicate"], r["object"]) for r in scene["relations"]}
    P_, S_, D_ = ids[1], ids[2], ids[3]
    assert (P_, "left_of", S_) in rel and (S_, "left_of", D_) in rel and (P_, "left_of", D_) in rel
    assert (S_, "above", D_) in rel                                      # centro silla y=185 < perro y=240
    assert all(r["predicate"] in vp.PREDICATES for r in scene["relations"])   # nada de 'inside'
    assert not any(r["predicate"] == "overlaps" for r in scene["relations"])  # no se tocan
    # la misma función que usa el motor da el mismo veredicto
    o = {x["id"]: x for x in scene["objects"]}
    assert vp.relation_holds(o[D_], o[P_], "right_of") and not vp.relation_holds(o[P_], o[D_], "right_of")


@test
def overlaps_con_iou():
    W = H = 100
    scene = vp.build_scene("i", W, H, [_mk_det(1, [0, 0, 50, 50], .9, W, H), _mk_det(2, [25, 25, 75, 75], .8, W, H)])
    assert any(r["predicate"] == "overlaps" for r in scene["relations"])    # IoU = 625/4375 = 0.14 > 0.10


@test
def tope_de_relaciones_respeta_esquema():
    rng = np.random.default_rng(3)
    objs = []
    for i in range(250):
        cx, cy = rng.uniform(5, 95, 2)
        objs.append({"id": f"obj_{i}", "centroid": [float(cx), float(cy)], "bbox": [cx - 1, cy - 1, cx + 1, cy + 1],
                     "confidence": 1 - i / 300})
    rels = vp.compute_relations(objs)
    assert 0 < len(rels) <= vp.MAX_RELATIONS
    assert len(vp.compute_relations(objs, include_inverse=True)) <= vp.MAX_RELATIONS


@test
def scene_valida_contra_esquema_oficial():
    schema = json.loads((Path(__file__).parent / "scene_schema.json").read_text())
    W, H = 200, 150
    scene = vp.build_scene("tpi_0001", W, H, [_mk_det(1, [10, 10, 90, 100], .9, W, H), _mk_det(2, [100, 60, 190, 140], .6, W, H)])
    assert vp.validate_scene(scene, schema) == []
    # rompemos cosas a propósito y el validador debe quejarse
    bad = json.loads(json.dumps(scene)); bad["objects"][0]["mask"]["counts"] = "abc"
    assert vp.validate_scene(bad, schema)
    bad = json.loads(json.dumps(scene)); bad["relations"].append({"subject": "obj_0", "predicate": "inside", "object": "obj_1"})
    assert vp.validate_scene(bad, schema)
    bad = json.loads(json.dumps(scene)); bad["objects"][0]["position"] = "left"
    assert vp.validate_scene(bad, schema)
    bad = json.loads(json.dumps(scene)); bad["objects"][0]["mask"]["counts"][0] += 1
    assert any("sum(counts)" in e for e in vp.validate_scene(bad))      # sin esquema: chequeo propio


@test
def geometria_desde_mascara():
    W = H = 100
    m = np.zeros((H, W), np.uint8); m[10:30, 10:30] = 1
    a = vp.object_attributes([0, 0, 100, 100], W, H, m, geometry_source="mask")
    assert abs(a["area_ratio"] - 0.04) < 1e-9 and a["centroid"] == [20.0, 20.0]
    b = vp.object_attributes([0, 0, 100, 100], W, H, m, geometry_source="bbox")
    assert b["area_ratio"] == 1.0


# ======================================================================
print("\n== FASE 4: motor determinista (10 preguntas sobre un JSON manual) ==")

NAME_TO_ID = {"person": 1, "chair": 2, "dog": 3, "cat": 4, "bench": 5}


def manual_scene():
    W = H = 300
    dets = [_mk_det(1, [10, 100, 60, 250], .95, W, H),      # persona izquierda
            _mk_det(2, [120, 120, 190, 250], .90, W, H),    # silla centro (grande: 70*130/90000=0.10)
            _mk_det(2, [130, 10, 160, 40], .50, W, H),      # otra silla, arriba (pequeña 0.01)
            _mk_det(3, [210, 200, 290, 280], .80, W, H)]    # perro
    return vp.build_scene("img", W, H, dets), {o["category_id"]: o["id"] for o in vp.build_scene("img", W, H, dets)["objects"][::-1]}


@test
def diez_preguntas():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)

    def ask(t, classes, text="", expect=None, **kw):
        q = {"question_id": "q", "type": t, "question": text, "class_en": classes}
        r = eng.answer(scene, q)
        assert r["normalized_answer"] == expect, (t, classes, text, r)
        assert len(r["evidence"]) == len(set(r["evidence"]))
        return r

    ask("count", ["chair"], expect=2)
    ask("count", ["cat"], expect=0)
    ask("existence", ["dog"], expect="yes")
    ask("existence", ["cat"], expect="no")
    ask("position", ["person"], expect="middle_left")
    ask("position", ["dog"], expect="bottom_right")
    ask("size", ["chair"], expect="large")                  # la silla más confiable es la grande
    ask("relation", ["person", "dog"], '¿La instancia de "persona" está a la izquierda de la instancia de "perro"?', "yes")
    ask("relation", ["dog", "chair"], '¿La instancia de "perro" está por encima de la instancia de "silla"?', "no")
    ask("relation", ["person", "cat"], '¿... a la derecha de ...?', "no")      # falta una clase -> 'no'
    ask("count", ["person"], expect=1)
    r = ask("description", [], expect=eng.answer(scene, {"question_id": "d", "type": "description", "question": "x", "class_en": []})["normalized_answer"])
    assert "4 objetos" in r["answer"]


@test
def parser_de_predicados():
    assert qe.parse_predicate('¿... está por debajo de ...?') == "below"
    assert qe.parse_predicate('¿... está por encima de ...?') == "above"
    assert qe.parse_predicate('¿... está a la derecha de ...?') == "right_of"
    assert qe.parse_predicate('¿... está a la izquierda de ...?') == "left_of"
    assert qe.parse_intent({"question": "¿Cuántas instancias de la categoría \"perro\"?", "class_en": ["dog"]})["type"] == "count"


@test
def count_entero_y_schema_de_pregunta():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)
    r = eng.answer(scene, {"question_id": 7, "type": "count", "question": "q", "class_en": ["chair"]})
    assert isinstance(r["normalized_answer"], int) and r["answer"] == "2" and r["question_id"] == "7"
    scene["questions"].append(r)
    schema = json.loads((Path(__file__).parent / "scene_schema.json").read_text())
    assert vp.validate_scene(scene, schema) == []


# ======================================================================
print("\n== FASE 5: LLM (con mock) ==")


def make_mock(intent_json=None, verbal=None, boom=False):
    def llm(prompt):
        if boom:
            raise RuntimeError("sin red")
        if prompt.startswith("Recibirás"):
            return intent_json if intent_json is not None else "{}"
        return verbal if verbal is not None else ""
    return llm


@test
def llm_intencion_valida_y_fiel():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)
    vocab = list(NAME_TO_ID)
    q = {"question_id": "Q1", "type": "count", "question": "¿Cuántas sillas hay?", "class_en": ["chair"]}
    llm = lw.LLMWrapper(make_mock('```json\n{"type":"count","subject":"chair","reference":null,"predicate":null}\n```',
                                  "Hay 2 sillas en la imagen."), eng, vocab)
    r = llm.answer(scene, q)
    assert r["normalized_answer"] == 2 and r["answer"] == "Hay 2 sillas en la imagen."
    assert r["interpreted_query"]["source"] == "llm"


@test
def llm_que_inventa_se_descarta():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)
    q = {"question_id": "Q1", "type": "count", "question": "¿Cuántas sillas?", "class_en": ["chair"]}
    llm = lw.LLMWrapper(make_mock('{"type":"count","subject":"chair"}', "Hay 5 sillas."), eng, list(NAME_TO_ID))
    r = llm.answer(scene, q)
    assert r["normalized_answer"] == 2 and "5" not in r["answer"] and "2" in r["answer"]   # usa la plantilla fiel
    assert llm.stats["verbal_fallback"] == 1


@test
def llm_json_roto_o_caido_usa_reglas():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)
    q = {"question_id": "Q1", "type": "existence", "question": "¿Hay al menos una instancia de \"perro\"?", "class_en": ["dog"]}
    for mock in (make_mock("no sé"), make_mock(boom=True), make_mock('{"type":"count","subject":"unicornio"}')):
        r = lw.LLMWrapper(mock, eng, list(NAME_TO_ID)).answer(scene, q)
        assert r["normalized_answer"] == "yes" and r["interpreted_query"]["source"] == "rules"
        assert r["answer"].startswith("Sí")


@test
def llm_relacion_y_fidelidad_si_no():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)
    q = {"question_id": "Q2", "type": "relation", "question": "¿La persona está a la izquierda del perro?", "class_en": ["person", "dog"]}
    js = '{"type":"relation","subject":"person","reference":"dog","predicate":"left_of"}'
    ok = lw.LLMWrapper(make_mock(js, "Sí, la persona está a la izquierda del perro."), eng, list(NAME_TO_ID)).answer(scene, q)
    assert ok["normalized_answer"] == "yes" and ok["answer"].startswith("Sí")
    bad = lw.LLMWrapper(make_mock(js, "No, no está."), eng, list(NAME_TO_ID))
    r = bad.answer(scene, q)
    assert r["answer"].startswith("Sí") and bad.stats["verbal_fallback"] == 1


@test
def schema_de_questions_llm():
    scene, _ = manual_scene()
    eng = qe.QueryEngine(NAME_TO_ID)
    q = {"question_id": "Q3", "type": "size", "question": "¿Qué tamaño tiene la silla?", "class_en": ["chair"]}
    r = lw.LLMWrapper(make_mock("{}"), eng, list(NAME_TO_ID)).answer(scene, q)
    scene["questions_llm"].append(r)
    schema = json.loads((Path(__file__).parent / "scene_schema.json").read_text())
    assert vp.validate_scene(scene, schema) == [], vp.validate_scene(scene, schema)


# ======================================================================
print(f"\n{len(PASSED)} pruebas OK")
