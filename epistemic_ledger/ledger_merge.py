"""`merge_obj`: deterministic evidence carry-forward over the per-turn ledgers (used for every number in the paper).

Why: the evaluator rebuilds the whole ledger each turn by re-emitting the JSON twice (obj pass, then per pass), and an
LLM does not copy prior fields faithfully: an `obj=True` established earlier can come back as `None`. The prompts already
say to keep earlier values; this function enforces it in code.

Rules, per (candidate, constraint), turn by turn:
  - obj is taken from the obj pass (`ledger_pairs[t][0]`), never from the per pass.
  - new obj True/False with evidence -> take it (True->False only on an explicit contradiction with evidence).
  - new obj None, or the constraint/candidate missing, or the obj pass malformed -> keep the previous obj/evidence.
  - candidates are matched across turns by normalized name (case, whitespace, dash/underscore and punctuation
    variants; else one name contained in the other).
  - per / per_evidence / status and the candidate set come from the per pass; a malformed per pass repeats the
    previous turn.

    from ledger_merge import merge_obj
    ledgers = merge_obj(ledger_data)          # list of per-turn ledgers, same shape as ledger_data["ledger"]
"""
import copy
import re
import unicodedata

_DASHES = re.compile(r"[‐-―−\-_]+")


def _norm(name):
    s = unicodedata.normalize("NFKC", str(name or "")).lower()
    s = _DASHES.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)   # punctuation variants ("X: Y" vs "X Y", quotes, dots)
    return " ".join(s.split())


def _has_evidence(ev):
    return ev is not None and str(ev).strip() not in ("", "None", "null")


def _resolve(name, state):
    """State key for a candidate name: exact normalized match, else containment (one name inside the other),
    else a new key."""
    n = _norm(name)
    if n in state:
        return n
    for k in state:
        short = min(len(n), len(k))
        if short >= 12 and (n in k or k in n):
            return k
    return n


def _cons(v):
    """(id, constraint) pairs; a malformed list-valued "constraints" is skipped like any malformed entry."""
    c = v.get("constraints")
    return c.items() if isinstance(c, dict) else ()


def merge_obj(ledger_data):
    per_turns = ledger_data.get("ledger") or []
    pairs = ledger_data.get("ledger_pairs") or []
    state = {}      # norm cand -> {"name": str, "status":..., "constraints": {ci: {...}}}
    out = []
    for t, per_led in enumerate(per_turns):
        obj_led = pairs[t][0] if t < len(pairs) and isinstance(pairs[t], (list, tuple)) and pairs[t] else None
        obj_led = obj_led if isinstance(obj_led, dict) else None
        per_led = per_led if isinstance(per_led, dict) else None

        # 1) per pass: status / per / per_evidence (keep previous entry when malformed or dropped)
        if per_led is not None:
            for cand, v in per_led.items():
                if not isinstance(v, dict):
                    continue
                key = _resolve(cand, state)
                ent = state.setdefault(key, {"name": cand, "status": None, "constraints": {}})
                ent["name"] = cand
                if "status" in v:
                    ent["status"] = v["status"]
                for ci, c in _cons(v):
                    if not isinstance(c, dict):
                        continue
                    cur = ent["constraints"].setdefault(ci, {"obj": None, "obj_evidence": None,
                                                             "per": None, "per_evidence": None})
                    for k in ("per", "per_evidence"):
                        if k in c:
                            cur[k] = c[k]
                    for k, val in c.items():   # keep any extra fields the per pass emits
                        if k not in ("obj", "obj_evidence", "per", "per_evidence"):
                            cur[k] = val

        # 2) obj pass: monotone merge of obj / obj_evidence
        if obj_led is not None:
            for cand, v in obj_led.items():
                if not isinstance(v, dict):
                    continue
                key = _resolve(cand, state)
                ent = state.setdefault(key, {"name": cand, "status": v.get("status"), "constraints": {}})
                for ci, c in _cons(v):
                    if not isinstance(c, dict):
                        continue
                    cur = ent["constraints"].setdefault(ci, {"obj": None, "obj_evidence": None,
                                                             "per": None, "per_evidence": None})
                    new, ev = c.get("obj"), c.get("obj_evidence")
                    if new is True and _has_evidence(ev):
                        cur["obj"], cur["obj_evidence"] = True, ev
                    elif new is False and _has_evidence(ev):
                        cur["obj"], cur["obj_evidence"] = False, ev
                    elif cur["obj"] is None and new in (True, False):
                        # no prior value: accept the LLM's value even without evidence (same as the raw ledger)
                        cur["obj"], cur["obj_evidence"] = new, ev
                    # new None / no evidence: keep the previous value

        # 3) emit: the candidate set is the per pass's (as in the raw ledger); a malformed per pass repeats the
        #    previous turn. obj/obj_evidence come from the merged state.
        if per_led is None:
            out.append(copy.deepcopy(out[-1]) if out else {})
            continue
        turn = {}
        for cand, v in per_led.items():
            if not isinstance(v, dict):
                continue
            ent = state[_resolve(cand, state)]
            turn[cand] = {"status": ent["status"],
                          "constraints": {ci: copy.deepcopy(ent["constraints"][ci])
                                          for ci, _ in _cons(v) if ci in ent["constraints"]}}
        out.append(turn)
    return out
