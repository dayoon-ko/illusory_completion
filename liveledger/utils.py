import json
from typing import Any, Dict, List, Tuple

ANSI_RESET = "\033[0m"
ANSI_USER = "\033[36m"
ANSI_THINK = "\033[31m"
ANSI_TOOL_CALL = "\033[33m"
ANSI_TOOL_MSG = "\033[35m"
ANSI_RESPONSE = "\033[32m"
ANSI_LEDGER = "\033[34m"
ANSI_PHASE = "\033[95m"


def parse_boxed(text: str) -> str:
    if not text:
        return ""
    boxed = r"\boxed"
    start = text.rfind(boxed)
    if start == -1:
        return text.strip()
    
    idx = start + len(boxed)
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != "{":
        return text.strip()
    
    depth = 1
    content_start = idx + 1
    idx += 1
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1
    
    if depth == 0:
        return text[content_start:idx-1].strip()
    return text.strip()


def log_event(label: str, text: str, color: str, entire: bool = False) -> None:
    if isinstance(text, bool):
        text = str(text)
    if not text or not str(text).strip():
        return
    print(f"{color}[{label}]{ANSI_RESET}")
    if entire:
        print(text)
    else:
        for line in str(text).splitlines()[:50]:
            print(f"  {line}")
    print()


class AgentStateMachine:
    """State machine for three-phase approach."""
    
    INIT = "INIT"
    SEARCH = "SEARCH"
    COMPLETE = "COMPLETE"
    
    def __init__(self):
        self.state = self.INIT
        self.last_tool = None
    
    def get_allowed_tools(self) -> List[str]:
        if self.state == self.INIT:
            return ["extract_constraints"]
        elif self.state == self.SEARCH:
            return ["search", "browse"]
        elif self.state == self.COMPLETE:
            return ["search", "browse"]
        return []
    
    def transition(self, tool_name: str, ledger_complete: bool = False) -> str:
        allowed = self.get_allowed_tools()
        
        if tool_name not in allowed:
            if self.state == self.INIT:
                return f"❌ You must call `extract_constraints` first. Allowed: {allowed}"
            else:
                return f"❌ Tool `{tool_name}` not allowed in state {self.state}. Allowed: {allowed}"
        
        self.last_tool = tool_name
        
        if tool_name == "extract_constraints":
            self.state = self.SEARCH
        elif tool_name in ["search", "browse"]:
            if ledger_complete:
                self.state = self.COMPLETE
        
        return ""
    
    def set_complete(self):
        self.state = self.COMPLETE
    
    def can_answer(self) -> bool:
        return self.state == self.COMPLETE
    
    def get_state_message(self) -> str:
        if self.state == self.INIT:
            return "📋 **State: INIT** - Extracting constraints..."
        elif self.state == self.SEARCH:
            return "🔍 **State: SEARCH** - Call `search` or `browse` to find evidence"
        elif self.state == self.COMPLETE:
            return "✅ **State: COMPLETE** - All constraints verified, you may answer with \\boxed{...}"
        return ""
    


class EpistemicLedger:
    """Manages the epistemic ledger."""
    
    def __init__(self):
        self.constraints: Dict[str, str] = {}
        self.ledger: Dict[str, Any] = {}
        self.stagnation_count = 0
    
    def set_constraints(self, constraint_list: List[str]) -> None:
        self.constraints = {f"C{i+1}": c for i, c in enumerate(constraint_list)}
        
    def get_stagnation_count(self) -> int:
        return self.stagnation_count
    
    def reset_stagnation_count(self) -> None:
        self.stagnation_count = 0
        
    def increase_stagnation_count(self) -> None:
        self.stagnation_count += 1
        
    def update(self, entries: List[Dict[str, Any]]) -> None:
        candidates = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            candidate = entry.get("candidate")
            if not candidate or candidate == "unknown":
                continue
                
            candidates.append(candidate)
            constraint = entry.get("constraint")
            obj = entry.get("obj")
            obj_evidence = entry.get("obj_evidence")
            
            if candidate not in self.ledger:
                self.ledger[candidate] = {
                    "constraints": {
                        cid: {"obj": None, "obj_evidence": None}
                        for cid in self.constraints
                    }
                }
            
            if constraint:
                if constraint not in self.ledger[candidate]["constraints"]:
                    self.ledger[candidate]["constraints"][constraint] = {"obj": None, "obj_evidence": None}
                
                if obj is not None and obj_evidence is not None:
                    self.ledger[candidate]["constraints"][constraint]["obj"] = obj
                    self.ledger[candidate]["constraints"][constraint]["obj_evidence"] = obj_evidence
        return candidates
        
    def format_constraints(self) -> str:
        if not self.constraints:
            return "(None yet)"
        return "\n".join(f"- **{cid}**: {desc}" for cid, desc in self.constraints.items())
    
    def format_constraints_for_update(self) -> str:
        if not self.constraints:
            return "[]"
        return "\n".join(f"- {cid}: \"{desc}\"" for cid, desc in self.constraints.items())
    
    def format_ledger(self, include_guidance=True) -> str:
        if not self.ledger:
            return "(Empty - no candidates yet)"
        
        lines = []
        feedbacks = []
        
        for candidate, data in self.ledger.items():
            lines.append(f"\n**{candidate}**")
            lines.append("| Constraint | obj | obj_evidence |")
            lines.append("|------------|-----|--------------|")
            
            for cid, cdata in data.get("constraints", {}).items():
                obj = cdata.get("obj")
                obj_evidence = cdata.get("obj_evidence") or "-"
                
                status_str = "true" if obj == True else ("false" if obj == False else "null")
                ev_short = (obj_evidence[:40] + "...") if len(str(obj_evidence)) > 40 else obj_evidence
                constraint_desc = self.constraints.get(cid, cid)
                lines.append(f"| {cid}: {constraint_desc} | {status_str} | {ev_short} |")
        
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
        if not self.ledger:
            return "{}"
        return json.dumps(self.ledger, indent=2)
    
    def check_completion_of_candidate(self, candidate: str) -> Tuple[bool, bool, List[str]]:
        
        is_false = False
        missing = []
        for cid, cdata in self.ledger[candidate].get("constraints", {}).items():
            if cdata.get("obj") is False:
                is_false = True
            elif cdata.get("obj") is not True:
                missing.append(cid)
        is_complete = not is_false and not missing
        return is_complete, is_false, missing
    
    def check_completion(self) -> Tuple[bool, List[str]]:
        is_complete = False
        missing = []
        for candidate in self.ledger.keys():
            is_complete, _, missing = self.check_completion_of_candidate(candidate)
            if is_complete:
                return True, missing 
            
        return False, missing
