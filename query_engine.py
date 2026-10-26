"""
query_engine.py  —  FASE 4: motor determinista. Sin torch, sin LLM.

  parse_intent(q)            pregunta -> intención estructurada (reglas por palabras clave)
  count / exist / position / size / relation / describe     funciones puras sobre objetos del SceneJSON
  QueryEngine.execute_intent(scene, intent) -> {answer, normalized_answer, evidence, confidence}
  QueryEngine.answer(scene, q)              -> registro listo para scene["questions"]

La intención tiene la forma {"type","subject","reference","predicate"} y es EXACTAMENTE
lo que el modo LLM devuelve como `interpreted_query`; así ambos modos comparten el motor.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Callable

from vision_pipeline import relation_holds

VALID_TYPES = ("description", "count", "existence", "position", "size", "relation")

# Orden importa: las frases más específicas primero.
RELATION_KEYWORDS: list[tuple[str, str]] = [
    ("por encima de", "above"), ("encima de", "above"), ("arriba de", "above"),
    ("por debajo de", "below"), ("debajo de", "below"), ("abajo de", "below"),
    ("a la derecha de", "right_of"), ("a la izquierda de", "left_of"),
    ("se superpone con", "overlaps"), ("se superpone", "overlaps"), ("solapa", "overlaps"),
]
TYPE_KEYWORDS: list[tuple[str, str]] = [   # solo si la pregunta no trae 'type'
    ("describ", "description"), ("cuantas", "count"), ("cuantos", "count"), ("cantidad", "count"),
    ("al menos una", "existence"), ("hay alguna", "existence"), ("existe", "existence"),
    ("en que zona", "position"), ("donde esta", "position"), ("posicion", "position"),
    ("tamano", "size"),
]


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def parse_predicate(question_text: str) -> str | None:
    q = _norm(question_text)
    for kw, pred in RELATION_KEYWORDS:
        if _norm(kw) in q:
            return pred
    return None


def parse_intent(q: dict) -> dict:
    """Paso 4.1. Usa q['type'] y q['class_en'] cuando existen; si no, intenta por palabras clave."""
    text = q.get("question", "")
    q_type = q.get("type")
    pred = parse_predicate(text)
    if q_type not in VALID_TYPES:
        q_type = next((t for kw, t in TYPE_KEYWORDS if kw in _norm(text)), None)
        if q_type is None and pred:
            q_type = "relation"
    classes = list(q.get("class_en") or [])
    return {"type": q_type, "subject": classes[0] if classes else None,
            "reference": classes[1] if len(classes) > 1 else None, "predicate": pred}


# ----------------------------------------------------------------------
# Funciones puras (Paso 4.2)
# ----------------------------------------------------------------------
def select(objects: list[dict], cat_ids: set[int]) -> list[dict]:
    """Instancias de las categorías pedidas, de mayor a menor confianza."""
    return sorted((o for o in objects if o["category_id"] in cat_ids), key=lambda o: -o["confidence"])


def count(objects, cat_ids) -> tuple[int, list[str]]:
    objs = select(objects, cat_ids)
    return len(objs), [o["id"] for o in objs]


def exist(objects, cat_ids) -> tuple[bool, list[str]]:
    objs = select(objects, cat_ids)
    return bool(objs), [o["id"] for o in objs]


def position(objects, cat_ids, fallback: str = "center") -> tuple[str, list[str], bool]:
    objs = select(objects, cat_ids)
    return (objs[0]["position"], [objs[0]["id"]], True) if objs else (fallback, [], False)


def size(objects, cat_ids, fallback: str = "medium") -> tuple[str, list[str], bool]:
    objs = select(objects, cat_ids)
    return (objs[0]["size"], [objs[0]["id"]], True) if objs else (fallback, [], False)


def relation(objects, ids_a, ids_b, predicate: str, policy: str = "top") -> tuple[bool, list[str]]:
    """
    policy='top': compara la instancia más confiable de A con la más confiable de B.
    policy='any': 'yes' si ALGÚN par (a, b) cumple el predicado.
    """
    A, B = select(objects, ids_a), select(objects, ids_b)
    if not A or not B:
        return False, []
    if policy == "top":
        a, b = A[0], B[0]
        return relation_holds(a, b, predicate), [a["id"], b["id"]]
    for a in A:
        for b in B:
            if a["id"] != b["id"] and relation_holds(a, b, predicate):
                return True, [a["id"], b["id"]]
    return False, [A[0]["id"], B[0]["id"]]


def describe(objects: list[dict], id_to_name: dict[int, str], top_k: int = 8) -> tuple[str, list[str]]:
    if not objects:
        return "No se detectaron objetos en la imagen.", []
    c = Counter(id_to_name.get(o["category_id"], str(o["category_id"])) for o in objects)
    parts = [f"{n} x {name}" for name, n in c.most_common(top_k)]
    extra = len(c) - top_k
    txt = (f"La imagen contiene {len(objects)} objetos detectados: " + ", ".join(parts)
           + (f" y {extra} categorías más." if extra > 0 else "."))
    return txt, [o["id"] for o in objects]


# ----------------------------------------------------------------------
# Motor
# ----------------------------------------------------------------------
class QueryEngine:
    def __init__(self, name_to_id: dict[str, int], relation_policy: str = "top",
                 position_fallback: str = "center", size_fallback: str = "medium"):
        self.name_to_id = name_to_id
        self.id_to_name = {v: k for k, v in name_to_id.items()}
        self.relation_policy = relation_policy
        self.position_fallback = position_fallback
        self.size_fallback = size_fallback
        self.handlers: dict[str, Callable[[dict, dict], dict]] = {
            "count": self._count, "existence": self._existence, "position": self._position,
            "size": self._size, "relation": self._relation, "description": self._description,
        }

    # -- helpers -------------------------------------------------------
    def _ids(self, name: str | None) -> set[int]:
        return {self.name_to_id[name]} if name in self.name_to_id else set()

    @staticmethod
    def _res(answer: Any, normalized: Any, evidence: list[str], objects: list[dict], found: bool = True) -> dict:
        ev = list(dict.fromkeys(evidence))                      # schema: uniqueItems
        conf = min((o["confidence"] for o in objects if o["id"] in ev), default=0.5) if found else 0.1
        return {"answer": str(answer), "normalized_answer": normalized, "evidence": ev,
                "confidence": float(min(1.0, max(0.0, conf)))}

    # -- handlers ------------------------------------------------------
    def _count(self, scene, it):
        n, ev = count(scene["objects"], self._ids(it["subject"]))
        return self._res(n, int(n), ev, scene["objects"])         # normalized_answer ENTERO

    def _existence(self, scene, it):
        ok, ev = exist(scene["objects"], self._ids(it["subject"]))
        a = "yes" if ok else "no"
        return self._res(a, a, ev, scene["objects"])

    def _position(self, scene, it):
        v, ev, found = position(scene["objects"], self._ids(it["subject"]), self.position_fallback)
        return self._res(v, v, ev, scene["objects"], found)

    def _size(self, scene, it):
        v, ev, found = size(scene["objects"], self._ids(it["subject"]), self.size_fallback)
        return self._res(v, v, ev, scene["objects"], found)

    def _relation(self, scene, it):
        pred = it.get("predicate")
        if not it.get("reference") or pred is None:
            return self._res("no", "no", [], scene["objects"], found=False)
        ok, ev = relation(scene["objects"], self._ids(it["subject"]), self._ids(it["reference"]),
                          pred, self.relation_policy)
        a = "yes" if ok else "no"
        return self._res(a, a, ev, scene["objects"], found=bool(ev))

    def _description(self, scene, it):
        txt, ev = describe(scene["objects"], self.id_to_name)
        return self._res(txt, txt, ev, scene["objects"])

    # -- API pública ---------------------------------------------------
    def execute_intent(self, scene: dict, intent: dict) -> dict:
        handler = self.handlers.get(intent.get("type"))
        if handler is None:
            return {"answer": "unsupported", "normalized_answer": "unsupported", "evidence": [], "confidence": 0.0}
        return handler(scene, intent)

    def answer(self, scene: dict, q: dict) -> dict:
        res = self.execute_intent(scene, parse_intent(q))
        return {"question_id": str(q["question_id"]), "type": str(q.get("type", "")),
                "question": str(q.get("question", "")), **res}
