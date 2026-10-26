"""test_extras.py — pruebas de detection_extras (sin GPU)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import dataset_utils as du
import detection_extras as de


def test_wbf_dos_modelos_coinciden():
    b1, b2 = [[10, 10, 50, 50]], [[14, 10, 54, 50]]
    b, s, l = de.weighted_box_fusion([b1, b2], [[0.8], [0.6]], [[1], [1]], iou_thr=0.5)
    assert len(b) == 1 and l[0] == 1
    assert abs(s[0] - 0.7) < 1e-9                            # media de scores, visto por 2 de 2 modelos
    assert 10 < b[0][0] < 14 and b[0][0] < 12.3              # más cerca de la caja de mayor score*peso
    assert abs(b[0][1] - 10) < 1e-9 and abs(b[0][3] - 50) < 1e-9


def test_wbf_caja_vista_por_un_solo_modelo_se_penaliza():
    b, s, l = de.weighted_box_fusion([[[0, 0, 10, 10]], [[100, 100, 120, 120]]], [[0.9], [0.9]], [[1], [1]])
    assert len(b) == 2 and np.allclose(s, 0.45)


def test_wbf_no_mezcla_etiquetas():
    b, s, l = de.weighted_box_fusion([[[0, 0, 10, 10]], [[0, 0, 10, 10]]], [[0.9], [0.9]], [[1], [2]])
    assert len(b) == 2


def test_wbf_vacio():
    b, s, l = de.weighted_box_fusion([np.zeros((0, 4)), np.zeros((0, 4))], [[], []], [[], []])
    assert len(b) == 0
    assert de.fuse_detections([[], []]) == []


def test_tiles_cubren_la_imagen():
    for (w, h, r, c) in [(640, 480, 2, 2), (425, 640, 3, 3), (224, 304, 1, 2)]:
        wins = de.tile_windows(w, h, r, c, 0.2)
        assert len(wins) == r * c
        cover = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in wins:
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            cover[y0:y1, x0:x1] = True
        assert cover.all(), (w, h, r, c)


def test_shift_dets():
    d = de.shift_dets([{"bbox": [1, 2, 3, 4], "score": .5, "label": 0}], 10, 20)
    assert d[0]["bbox"] == [11, 22, 13, 24]


def test_federated_report_suma_igual_que_evaluate_dataset():
    rng = np.random.default_rng(5)
    gts, preds = {}, {}
    for k in range(30):
        info = {"width": 100, "height": 100, "neg_category_ids": [int(x) for x in rng.choice([4, 5, 6], 1)],
                "not_exhaustive_category_ids": [1] if rng.random() < .4 else []}
        anns = []
        for c in (1, 2, 3):
            if rng.random() < .7:
                x, y = rng.uniform(0, 60, 2); anns.append({"category_id": c, "bbox": [x, y, 20, 20]})
        gts[k] = du.prepare_gt(info, anns)
        ps = []
        for a in gts[k]["anns"]:
            ps.append({"category_id": a["category_id"], "bbox": [v + rng.normal(0, 3) for v in a["bbox_xyxy"]], "score": float(rng.random())})
        for _ in range(5):
            x, y = rng.uniform(0, 70, 2)
            ps.append({"category_id": int(rng.integers(1, 8)), "bbox": [x, y, x + 15, y + 15], "score": float(rng.random())})
        preds[k] = ps
    ref = du.evaluate_dataset(preds, gts)
    df = de.federated_report(preds, gts, {})
    assert (df.TP.sum(), df.FP_counted.sum(), df.FN.sum()) == (ref["tp"], ref["fp"], ref["fn"])
    assert df.preds_ignored.sum() == ref["ignored_preds"]


def test_refine_mask():
    m = np.zeros((60, 60), np.uint8); m[10:40, 10:40] = 1
    noisy = m.copy(); noisy[50, 50] = 1; noisy[52:54, 52:54] = 1            # motas aisladas
    r = de.refine_mask(noisy, open_k=3, close_k=3)
    assert r[50, 50] == 0 and r[10:40, 10:40].all()
    ring = np.zeros((60, 60), np.uint8); ring[10:50, 10:50] = 1; ring[20:40, 20:40] = 0   # aro: el agujero debe sobrevivir
    assert de.refine_mask(ring)[30, 30] == 0
    tiny = np.zeros((20, 20), np.uint8); tiny[5, 5:8] = 1                   # se borraría con la apertura -> devuelve original
    assert np.array_equal(de.refine_mask(tiny), tiny)
    clipped = de.refine_mask(m, box=[10, 10, 25, 40], open_k=1, close_k=1, min_keep_ratio=0.0)
    assert clipped[:, 25:].sum() == 0


def test_score_submission_df_ignora_no_verificadas_y_penaliza_negativas():
    import json, pandas as pd
    import experiments as ex, vision_pipeline as vp
    from test_integration import make_dev, IMGS, VOCAB
    dev = make_dev()
    def scene_for(i, extra):
        dets = []
        for c, b in IMGS[i]:
            m = np.zeros((120, 160), np.uint8); m[b[1]:b[3], b[0]:b[2]] = 1
            dets.append({"category_id": c, "bbox": [float(v) for v in b], "score": .9, "mask": m})
        for c, b in extra:
            m = np.zeros((120, 160), np.uint8); m[b[1]:b[3], b[0]:b[2]] = 1
            dets.append({"category_id": c, "bbox": [float(v) for v in b], "score": .5, "mask": m})
        return json.dumps(vp.build_scene(f"tpi_{i:04d}", 160, 120, dets))
    # imagen 2: 'dog'(3) es negativa verificada; imagen 3: 'person'(1) es negativa. Imagen 1: persona extra en categoría POSITIVA exhaustiva.
    rows = [{"image_id": "tpi_0001", "SceneJSON": scene_for(1, [(1, [100, 5, 140, 40])])},          # FP que cuenta (person exhaustiva)
            {"image_id": "tpi_0002", "SceneJSON": scene_for(2, [(3, [5, 5, 30, 30])])},               # FP que cuenta (dog negativa)
            {"image_id": "tpi_0003", "SceneJSON": scene_for(3, [])}]
    r = ex.score_submission_df(pd.DataFrame(rows), dev)
    assert r["fp"] == 2 and r["fn"] == 0 and r["tp"] == 8
    r2 = ex.score_submission_df(pd.DataFrame(rows), dev, kind="mask")
    assert (r2["tp"], r2["fp"], r2["fn"]) == (8, 2, 0)


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); n += 1; print("  OK ", name)
    print(f"\n{n} pruebas OK")


