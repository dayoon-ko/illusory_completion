"""The live ledger the agent sees: one row per (candidate, constraint), obj = true / false / null + a quoted evidence."""
import json
from typing import Any, Dict, List, Tuple


def parse_boxed(text: str) -> str:
    """Content of the last \\boxed{...}; the whole text if there is none."""
    if not text:
        return ""
    start = text.rfind(r"\boxed")
    if start == -1:
        return text.strip()
    idx = start + len(r"\boxed")
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != "{":
        return text.strip()
    depth, content_start = 1, idx + 1
    idx += 1
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1
    return text[content_start:idx - 1].strip() if depth == 0 else text.strip()


class EpistemicLedger:
    def __init__(self):
        self.constraints: Dict[str, str] = {}
        self.ledger: Dict[str, Any] = {}

    def set_constraints(self, constraint_list: List[str]) -> None:
        self.constraints = {f"C{i+1}": c for i, c in enumerate(constraint_list)}

    def update(self, entries: List[Dict[str, Any]]) -> List[str]:
        candidates = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            candidate = entry.get("candidate")
            if not candidate or candidate == "unknown":
                continue
            candidates.append(candidate)
            constraint, obj, obj_evidence = entry.get("constraint"), entry.get("obj"), entry.get("obj_evidence")
            if candidate not in self.ledger:
                self.ledger[candidate] = {"constraints": {cid: {"obj": None, "obj_evidence": None} for cid in self.constraints}}
            if constraint:
                cons = self.ledger[candidate]["constraints"]
                if constraint not in cons:
                    cons[constraint] = {"obj": None, "obj_evidence": None}
                if obj is not None and obj_evidence is not None:
                    cons[constraint]["obj"] = obj
                    cons[constraint]["obj_evidence"] = obj_evidence
        return candidates

    def format_constraints_for_update(self) -> str:
        if not self.constraints:
            return "[]"
        return "\n".join(f"- {cid}: \"{desc}\"" for cid, desc in self.constraints.items())

    def format_ledger(self, include_guidance=True) -> str:
        """The markdown table shown to the agent (+ one 'Feedbacks' line per candidate unless include_guidance=False)."""
        if not self.ledger:
            return "(Empty - no candidates yet)"
        lines, feedbacks = [], []
        for candidate, data in self.ledger.items():
            lines.append(f"\n**{candidate}**")
            lines.append("| Constraint | obj | obj_evidence |")
            lines.append("|------------|-----|--------------|")
            for cid, cdata in data.get("constraints", {}).items():
                obj = cdata.get("obj")
                obj_evidence = cdata.get("obj_evidence") or "-"
                status_str = "true" if obj == True else ("false" if obj == False else "null")
                ev_short = (obj_evidence[:40] + "...") if len(str(obj_evidence)) > 40 else obj_evidence
                lines.append(f"| {cid}: {self.constraints.get(cid, cid)} | {status_str} | {ev_short} |")
            if include_guidance:
                is_complete, is_false, missing = self.check_completion_of_candidate(candidate)
                if is_complete:
                    feedbacks.append(f"{candidate}: All constraints verified - this candidate is a valid answer!")
                elif is_false:
                    feedbacks.append(f"{candidate}: This candidate is not a valid answer because it is false for some constraints")
                else:
                    feedbacks.append(f"{candidate}: You need to verify {', '.join(missing) if missing else 'no active constraints'} to be a valid answer!")
        return "\n".join(lines) + "\n\nFeedbacks:\n" + "\n".join(feedbacks)

    def format_ledger_json(self) -> str:
        return json.dumps(self.ledger, indent=2) if self.ledger else "{}"

    def check_completion_of_candidate(self, candidate: str) -> Tuple[bool, bool, List[str]]:
        is_false, missing = False, []
        for cid, cdata in self.ledger[candidate].get("constraints", {}).items():
            if cdata.get("obj") is False:
                is_false = True
            elif cdata.get("obj") is not True:
                missing.append(cid)
        return not is_false and not missing, is_false, missing
