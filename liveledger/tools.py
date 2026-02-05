from __future__ import annotations
from typing import Any, Dict, List

# ============================================================================
# TOOLS - Separated by phase
# ============================================================================

TOOLS_EXTRACT: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "extract_constraints",
            "description": (
                "Parse the question into atomic, verifiable constraints.\n\n"
                "Each constraint should be a single, independently verifiable condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of atomic constraints the answer must satisfy"
                    }
                },
                "required": ["constraints"]
            }
        }
    }
]

TOOLS_SEARCH: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information. Use this to gather evidence for verifying constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search queries"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Read web page content for detailed information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs to read"
                    }
                },
                "required": ["urls"]
            }
        }
    },
]

TOOLS_UPDATE: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_ledger",
            "description": (
                "Record evidence in the ledger based on search results.\n\n"
                "Each entry records evidence for ONE constraint of ONE candidate.\n"
                "Only include entries where new evidence was found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "description": "List of evidence entries to add or update",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate": {
                                    "type": "string",
                                    "description": "The candidate answer being evaluated (e.g., 'Sony')"
                                },
                                "constraint": {
                                    "type": "string",
                                    "description": "Constraint identifier (e.g., 'C1', 'C2')"
                                },
                                "obj": {
                                    "type": ["boolean", "null"],
                                    "description": "true if evidence PROVES the constraint is satisfied, false if evidence DISPROVES it, null if inconclusive"
                                },
                                "obj_evidence": {
                                    "type": ["string", "null"],
                                    "description": "Verbatim quote from search results supporting the status, or null if no relevant quote found"
                                }
                            },
                            "required": ["candidate", "constraint", "obj", "obj_evidence"]
                        }
                    }
                },
                "required": ["entries"]
            }
        }
    }
]
