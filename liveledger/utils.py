"""
Utility functions and classes for the epistemic agent.

Includes:
- Color constants for terminal output
- Parsing utilities
- Logging utilities
- Agent state machine
- Epistemic ledger
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


# ============================================================================
# ANSI COLOR CONSTANTS
# ============================================================================

ANSI_RESET = "\033[0m"
ANSI_USER = "\033[36m"      # Cyan
ANSI_THINK = "\033[31m"     # Red
ANSI_TOOL_CALL = "\033[33m" # Yellow
ANSI_TOOL_MSG = "\033[35m"  # Magenta
ANSI_RESPONSE = "\033[32m"  # Green
ANSI_LEDGER = "\033[34m"    # Blue
ANSI_PHASE = "\033[95m"     # Bright Magenta


# ============================================================================
# PARSING UTILITIES
# ============================================================================

def parse_boxed(text: str) -> str:
    """
    Extract content from LaTeX \\boxed{...} notation.
    
    Handles nested braces correctly by tracking brace depth.
    Falls back to returning the full text if no \\boxed{} found.
    
    Args:
        text: Text that may contain \\boxed{content}
        
    Returns:
        Content inside the last \\boxed{}, or the original text if not found
        
    Example:
        >>> parse_boxed("The answer is \\boxed{42}")
        "42"
        >>> parse_boxed("\\boxed{A = {x, y}}")
        "A = {x, y}"
    """
    if not text:
        return ""
    
    boxed = r"\boxed"
    start = text.rfind(boxed)
    
    if start == -1:
        return text.strip()
    
    # Skip whitespace after \boxed
    idx = start + len(boxed)
    while idx < len(text) and text[idx].isspace():
        idx += 1
    
    # Must have opening brace
    if idx >= len(text) or text[idx] != "{":
        return text.strip()
    
    # Track brace depth to handle nested braces
    depth = 1
    content_start = idx + 1
    idx += 1
    
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1
    
    # Successfully found matching closing brace
    if depth == 0:
        return text[content_start:idx-1].strip()
    
    return text.strip()


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_event(label: str, text: str, color: str, entire: bool = False) -> None:
    """
    Log an event with colored label and optional truncation.
    
    Args:
        label: Label for the event (e.g., "USER", "THINKING")
        text: Content to log
        color: ANSI color code for the label
        entire: If False, truncate to first 50 lines. If True, show all.
        
    Example:
        >>> log_event("USER", "What is 2+2?", ANSI_USER)
        [USER]
          What is 2+2?
    """
    # Handle non-string inputs
    if isinstance(text, bool):
        text = str(text)
    
    # Skip empty content
    if not text or not str(text).strip():
        return
    
    # Print colored label
    print(f"{color}[{label}]{ANSI_RESET}")
    
    # Print content (with optional truncation)
    if entire:
        print(text)
    else:
        for line in str(text).splitlines()[:50]:
            print(f"  {line}")
    
    print()


# ============================================================================
# AGENT STATE MACHINE
# ============================================================================

class AgentStateMachine:
    """
    State machine for three-phase epistemic agent.
    
    States:
        INIT: Initial state, must extract constraints
        SEARCH: Can search/browse for evidence
        COMPLETE: All constraints verified, can provide answer
    
    Transitions:
        INIT -> SEARCH: After extract_constraints
        SEARCH -> COMPLETE: When ledger is complete
        COMPLETE: Remains in COMPLETE (can still search/browse)
    """
    
    # State constants
    INIT = "INIT"
    SEARCH = "SEARCH"
    COMPLETE = "COMPLETE"
    
    def __init__(self):
        """Initialize state machine in INIT state."""
        self.state = self.INIT
        self.last_tool = None
    
    def get_allowed_tools(self) -> List[str]:
        """
        Get list of tools allowed in current state.
        
        Returns:
            List of allowed tool names
        """
        if self.state == self.INIT:
            return ["extract_constraints"]
        elif self.state == self.SEARCH:
            return ["search", "browse"]
        elif self.state == self.COMPLETE:
            return ["search", "browse"]
        return []
    
    def transition(self, tool_name: str, ledger_complete: bool = False) -> str:
        """
        Attempt to transition based on tool call.
        
        Args:
            tool_name: Name of tool being called
            ledger_complete: Whether the ledger is complete (all constraints verified)
            
        Returns:
            Error message if transition invalid, empty string if valid
        """
        allowed = self.get_allowed_tools()
        
        # Validate tool is allowed
        if tool_name not in allowed:
            if self.state == self.INIT:
                return (
                    f"❌ You must call `extract_constraints` first. "
                    f"Allowed: {allowed}"
                )
            else:
                return (
                    f"❌ Tool `{tool_name}` not allowed in state {self.state}. "
                    f"Allowed: {allowed}"
                )
        
        # Record tool call
        self.last_tool = tool_name
        
        # Perform state transitions
        if tool_name == "extract_constraints":
            self.state = self.SEARCH
        elif tool_name in ["search", "browse"]:
            if ledger_complete:
                self.state = self.COMPLETE
        
        return ""
    
    def set_complete(self):
        """Manually set state to COMPLETE (called when ledger becomes complete)."""
        self.state = self.COMPLETE
    
    def can_answer(self) -> bool:
        """
        Check if agent can provide final answer.
        
        Returns:
            True if in COMPLETE state
        """
        return self.state == self.COMPLETE
    
    def get_state_message(self) -> str:
        """
        Get human-readable state message.
        
        Returns:
            Description of current state and allowed actions
        """
        if self.state == self.INIT:
            return "📋 **State: INIT** - Extracting constraints..."
        elif self.state == self.SEARCH:
            return "🔍 **State: SEARCH** - Call `search` or `browse` to find evidence"
        elif self.state == self.COMPLETE:
            return "✅ **State: COMPLETE** - All constraints verified, you may answer with \\boxed{...}"
        return ""


# ============================================================================
# EPISTEMIC LEDGER
# ============================================================================

class EpistemicLedger:
    """
    Manages the epistemic ledger for tracking evidence.
    
    The ledger tracks:
    - Constraints: Atomic conditions the answer must satisfy
    - Candidates: Potential answers being evaluated
    - Evidence: For each (candidate, constraint) pair, track verification status and evidence
    
    Structure:
        constraints: Dict[constraint_id, constraint_description]
        ledger: Dict[candidate, Dict[constraint_id, {obj: bool|None, obj_evidence: str|None}]]
    """
    
    def __init__(self):
        """Initialize empty ledger."""
        self.constraints: Dict[str, str] = {}
        self.ledger: Dict[str, Any] = {}
        self.stagnation_count = 0
    
    def set_constraints(self, constraint_list: List[str]) -> None:
        """
        Initialize constraints from extracted list.
        
        Args:
            constraint_list: List of constraint descriptions
        """
        self.constraints = {f"C{i+1}": c for i, c in enumerate(constraint_list)}
    
    def get_stagnation_count(self) -> int:
        """Get number of consecutive updates with no progress."""
        return self.stagnation_count
    
    def reset_stagnation_count(self) -> None:
        """Reset stagnation counter (called when progress is made)."""
        self.stagnation_count = 0
    
    def increase_stagnation_count(self) -> None:
        """Increment stagnation counter (called when no progress made)."""
        self.stagnation_count += 1
    
    def update(self, entries: List[Dict[str, Any]]) -> List[str]:
        """
        Update ledger with new evidence entries.
        
        Args:
            entries: List of evidence entries, each with:
                - candidate: Candidate answer name
                - constraint: Constraint ID (e.g., "C1")
                - obj: Verification status (true/false/null)
                - obj_evidence: Evidence text
        
        Returns:
            List of candidate names that were updated
        """
        candidates = []
        
        for entry in entries:
            candidate = entry.get("candidate")
            
            # Skip invalid candidates
            if not candidate or candidate == "unknown":
                continue
            
            candidates.append(candidate)
            constraint = entry.get("constraint")
            obj = entry.get("obj")
            obj_evidence = entry.get("obj_evidence")
            
            # Initialize candidate if not exists
            if candidate not in self.ledger:
                self.ledger[candidate] = {
                    "constraints": {
                        cid: {"obj": None, "obj_evidence": None}
                        for cid in self.constraints
                    }
                }
            
            # Update constraint evidence
            if constraint:
                if constraint not in self.ledger[candidate]["constraints"]:
                    self.ledger[candidate]["constraints"][constraint] = {
                        "obj": None,
                        "obj_evidence": None
                    }
                
                if obj is not None and obj_evidence is not None:
                    self.ledger[candidate]["constraints"][constraint]["obj"] = obj
                    self.ledger[candidate]["constraints"][constraint]["obj_evidence"] = obj_evidence
        
        return candidates
    
    def format_constraints(self) -> str:
        """
        Format constraints for display.
        
        Returns:
            Human-readable list of constraints
        """
        if not self.constraints:
            return "(None yet)"
        return "\n".join(f"- **{cid}**: {desc}" for cid, desc in self.constraints.items())
    
    def format_constraints_for_update(self) -> str:
        """
        Format constraints for update prompt.
        
        Returns:
            List of constraints with IDs
        """
        if not self.constraints:
            return "[]"
        return "\n".join(f"- {cid}: \"{desc}\"" for cid, desc in self.constraints.items())
    
    def format_ledger(self) -> str:
        """
        Format ledger as markdown tables with feedback.
        
        Returns:
            Human-readable ledger with verification status for each candidate
        """
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
                
                # Format status
                status_str = "true" if obj is True else ("false" if obj is False else "null")
                
                # Truncate evidence for display
                ev_short = obj_evidence[:40] + "..." if len(str(obj_evidence)) > 40 else obj_evidence
                
                # Get constraint description
                constraint_desc = self.constraints.get(cid, cid)
                
                lines.append(f"| {cid}: {constraint_desc} | {status_str} | {ev_short} |")
            
            # Generate feedback for this candidate
            is_complete, is_false, missing = self.check_completion_of_candidate(candidate)
            if is_complete:
                feedbacks.append(
                    f"{candidate}: All constraints verified - this candidate is a valid answer!"
                )
            elif is_false:
                feedbacks.append(
                    f"{candidate}: This candidate is not a valid answer because "
                    f"it is false for some constraints"
                )
            else:
                missing_str = ', '.join(missing) if missing else 'no active constraints'
                feedbacks.append(
                    f"{candidate}: You need to verify {missing_str} to be a valid answer!"
                )
        
        return "\n".join(lines) + "\n\nFeedbacks:\n" + "\n".join(feedbacks)
    
    def format_ledger_json(self) -> str:
        """
        Format ledger as JSON string.
        
        Returns:
            JSON representation of ledger
        """
        if not self.ledger:
            return "{}"
        return json.dumps(self.ledger, indent=2)
    
    def check_completion_of_candidate(self, candidate: str) -> Tuple[bool, bool, List[str]]:
        """
        Check if a candidate has all constraints verified.
        
        Args:
            candidate: Candidate name to check
            
        Returns:
            Tuple of (is_complete, is_false, missing_constraints)
            - is_complete: True if all constraints are verified as true
            - is_false: True if any constraint is verified as false
            - missing_constraints: List of constraint IDs still unverified
        """
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
        """
        Check if any candidate has all constraints verified.
        
        Returns:
            Tuple of (is_complete, missing_constraints)
            - is_complete: True if at least one candidate is fully verified
            - missing_constraints: List of missing constraints for candidates
        """
        for candidate in self.ledger.keys():
            is_complete, _, missing = self.check_completion_of_candidate(candidate)
            if is_complete:
                return True, []
        
        # Return missing constraints from first candidate as reference
        if self.ledger:
            first_candidate = next(iter(self.ledger.keys()))
            _, _, missing = self.check_completion_of_candidate(first_candidate)
            return False, missing
        
        return False, []