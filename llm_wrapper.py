"""
llm_wrapper.py  —  FASE 5: el LLM traduce y verbaliza; NO razona sobre la imagen.

    interpret(q)   : pregunta en español -> intención JSON (validada; si falla, fallback por reglas)
    verbalize(...) : respuesta del motor -> frase natural (aceptada solo si es FIEL al motor)
    answer(scene,q): registro para scene["questions_llm"] (incluye `interpreted_query`)

El LLM se inyecta como un callable `llm_fn(prompt: str) -> str`; así se prueba con un mock
y se puede cambiar de proveedor (modelo local de HF, API, etc.) sin tocar la lógica.
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Callable

from query_engine import VALID_TYPES, QueryEngine, parse_intent

POSITION_ES = {"top_left": "arriba a la izquierda", "top_center": "arriba al centro", "top_right": "arriba a la derecha",
               "middle_left": "en el medio a la izquierda", "center": "en el centro", "middle_right": "en el medio a la derecha",
               "bottom_left": "abajo a la izquierda", "bottom_center": "abajo al centro", "bottom_right": "abajo a la derecha"}
SIZE_ES = {"small": "pequeño", "medium": "mediano", "large": "grande"}
PREDICATES = ("left_of", "right_of", "above", "below", "overlaps")

INTENT_PROMPT = """Recibirás una pregunta en español sobre una imagen. Devuelve SOLO un JSON (sin texto extra, sin markdown) con la intención:
{{"type": "count|existence|position|size|relation|description", "subject": "<categoría>", "reference": "<categoría o null>", "predicate": "<predicado o null>"}}

Reglas:
- "subject" y "reference" deben ser EXACTAMENTE un nombre de esta lista de categorías (en inglés): {vocab}
- "reference" y "predicate" solo se usan en type=relation (la pregunta compara subject con reference).
- "predicate" es uno de: left_of, right_of, above, below, overlaps. ("por encima de"=above, "por debajo de"=below, "a la derecha de"=right_of, "a la izquierda de"=left_of)
- Para description deja subject, reference y predicate en null.

Pregunta: {question}
JSON:"""

VERBALIZE_PROMPT = """El motor determinista calculó esta respuesta: {answer}
Basándote SOLO en este resumen de la escena: {summary}
redacta una respuesta natural y concisa (una frase, en español) a la pregunta del usuario.
No agregues información que no esté en la respuesta del motor ni en el resumen.
Incluye el dato exacto: {must_contain}

Pregunta: {question}
Respuesta:"""


def extract_json(text: str) -> dict | None:
    """Primer objeto JSON dentro del texto (tolera ```json ... ``` y texto alrededor)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def resolve_category(name: str | None, vocab_names: list[str], cutoff: float = 0.8) -> str | None:
    if not name or str(name).lower() in ("null", "none"):
        return None
    n = str(name).strip().lower()
    lower = {v.lower(): v for v in vocab_names}
    if n in lower:
        return lower[n]
    m = difflib.get_close_matches(n, list(lower), n=1, cutoff=cutoff)
    return lower[m[0]] if m else None


def validate_intent(raw: dict | None, vocab_names: list[str]) -> dict | None:
    """Devuelve la intención normalizada o None si es inválida."""
    if not raw or raw.get("type") not in VALID_TYPES:
        return None
    t = raw["type"]
    subject = resolve_category(raw.get("subject"), vocab_names)
    reference = resolve_category(raw.get("reference"), vocab_names)
    pred = raw.get("predicate") if raw.get("predicate") in PREDICATES else None
    if t != "description" and subject is None:
        return None
    if t == "relation" and (reference is None or pred is None):
        return None
    return {"type": t, "subject": subject, "reference": reference if t == "relation" else None,
            "predicate": pred if t == "relation" else None}


def is_faithful(text: str, intent_type: str, normalized) -> bool:
    """El texto del LLM debe contener el dato que calculó el motor; si no, se descarta."""
    if not text or not text.strip():
        return False
    low = text.lower()
    if intent_type == "count":
        return re.search(rf"(?<!\d){int(normalized)}(?!\d)", text) is not None
    if intent_type in ("existence", "relation"):
        yes = re.search(r"\bs[ií]\b", low) is not None
        no = re.search(r"\bno\b", low) is not None
        return (yes and not no) if normalized == "yes" else (no and not yes)
    if intent_type == "position":
        return normalized in text or POSITION_ES.get(normalized, "\0") in low
    if intent_type == "size":
        return normalized in low or SIZE_ES.get(normalized, "\0") in low
    return True      # description: texto libre


def template_answer(intent: dict, normalized, engine_answer: str) -> str:
    """Verbalización sin LLM (fallback): siempre fiel al motor."""
    t, s, r = intent["type"], intent.get("subject"), intent.get("reference")
    if t == "count":
        return f"Hay {normalized} instancia(s) de '{s}' en la imagen."
    if t == "existence":
        return f"Sí, hay al menos una instancia de '{s}'." if normalized == "yes" else f"No, no se detecta ninguna instancia de '{s}'."
    if t == "position":
        return f"La instancia de '{s}' está {POSITION_ES.get(normalized, normalized)}."
    if t == "size":
        return f"La instancia de '{s}' tiene un tamaño {SIZE_ES.get(normalized, normalized)}."
    if t == "relation":
        return (f"Sí, '{s}' cumple la relación '{intent.get('predicate')}' respecto de '{r}'." if normalized == "yes"
                else f"No, '{s}' no cumple la relación '{intent.get('predicate')}' respecto de '{r}'.")
    return engine_answer


def scene_summary(scene: dict, id_to_name: dict[int, str], max_items: int = 15) -> str:
    objs = sorted(scene["objects"], key=lambda o: -o["confidence"])[:max_items]
    return json.dumps([{"id": o["id"], "category": id_to_name.get(o["category_id"]), "position": o["position"],
                        "size": o["size"]} for o in objs], ensure_ascii=False)


class LLMWrapper:
    def __init__(self, llm_fn: Callable[[str], str], engine: QueryEngine, vocab_names: list[str],
                 use_llm_intent: bool = True, use_llm_verbalization: bool = True):
        self.llm_fn, self.engine, self.vocab_names = llm_fn, engine, vocab_names
        self.use_llm_intent, self.use_llm_verbalization = use_llm_intent, use_llm_verbalization
        self.stats = {"intent_llm_ok": 0, "intent_fallback": 0, "verbal_llm_ok": 0, "verbal_fallback": 0}

    def interpret(self, q: dict) -> dict:
        if self.use_llm_intent:
            try:
                prompt = INTENT_PROMPT.format(vocab=", ".join(self.vocab_names), question=q.get("question", ""))
                intent = validate_intent(extract_json(self.llm_fn(prompt)), self.vocab_names)
            except Exception:
                intent = None
            if intent:
                self.stats["intent_llm_ok"] += 1
                return {**intent, "source": "llm"}
        self.stats["intent_fallback"] += 1
        return {**parse_intent(q), "source": "rules"}

    def verbalize(self, q: dict, intent: dict, res: dict, scene: dict) -> str:
        if self.use_llm_verbalization:
            try:
                must = res["normalized_answer"] if intent["type"] != "description" else "(libre)"
                prompt = VERBALIZE_PROMPT.format(answer=res["answer"], must_contain=must,
                                                 summary=scene_summary(scene, self.engine.id_to_name),
                                                 question=q.get("question", ""))
                text = self.llm_fn(prompt).strip()
                if is_faithful(text, intent["type"], res["normalized_answer"]):
                    self.stats["verbal_llm_ok"] += 1
                    return text
            except Exception:
                pass
        self.stats["verbal_fallback"] += 1
        return template_answer(intent, res["normalized_answer"], res["answer"])

    def answer(self, scene: dict, q: dict) -> dict:
        intent = self.interpret(q)
        res = self.engine.execute_intent(scene, intent)        # <- el cálculo es SIEMPRE determinista
        return {"question_id": str(q["question_id"]), "type": str(q.get("type", intent["type"] or "")),
                "question": str(q.get("question", "")),
                "answer": self.verbalize(q, intent, res, scene),
                "normalized_answer": res["normalized_answer"], "evidence": res["evidence"],
                "confidence": res["confidence"], "interpreted_query": intent}
