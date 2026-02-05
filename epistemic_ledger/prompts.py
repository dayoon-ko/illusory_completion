prompt_checklist_generation = """Please extract explicit, externally verifiable constraints from a question that an answer must satisfy.

Your task is to read a question and produce a constraint list in JSON format. 

Each constraint item must describe a condition that the answer must satisfy and must be objectively verifiable using an external source (e.g., factual records or publicly available information).

## Instructions

1. Decompose the question into a list of atomic constraints that cannot be further decomposed into smaller conditions. 
2. For each constraint, 
   - It must be **explicitly mentioned** in the question.
   - It must be objectively verifiable using an external source (e.g., factual records or publicly available information).
   - **Exclude** implicit, assumed, or inferred constraints or constraints about the **answer format** (e.g., length, structure, naming style).
3. Output **only** valid JSON in the specified format.

---

## JSON Format
```json
{{
  "constraint_1": {{ "constraint": "<externally verifiable condition the answer must satisfy>" }},
  "constraint_2": {{ "constraint": "<externally verifiable condition the answer must satisfy>" }}
}}
```

---

## Examples


**Example 1**

Question: Name a publicly traded technology company that was founded before 1990, is headquartered in Japan, and is listed on the Tokyo Stock Exchange.

Output:
```json
{{
    "constraint_2": {{"constraint": "The company must be a publicly traded technology company"}},
    "constraint_3": {{"constraint": "The company must have been founded before 1990"}},
    "constraint_4": {{"constraint": "The company must be headquartered in Japan"}},
    "constraint_5": {{"constraint": "The company must be listed on the Tokyo Stock Exchange"}}
}}
```

**Example 2**

Question: Name a film that won the Academy Award for Best Picture, was directed by a woman, and was released in the 2010s.

Output:
```json
{{
    "constraint_1": {{"constraint": "The film must have won the Academy Award for Best Picture"}},
    "constraint_2": {{"constraint": "The film must have been directed by a woman"}},
    "constraint_3": {{"constraint": "The film must have been released released in the 2010s"}}
}}
```

**Example 3**

Question: Name an Olympic athlete who has won gold medals in at least three different Summer Olympic Games and competed for the United States.

Output:
```json
{{
    "constraint_1": {{"constraint": "The person must be an Olympic athlete"}},
    "constraint_2": {{"constraint": "The person must have won gold medals in at least three different Summer Olympic Games"}},
    "constraint_3": {{"constraint": "The person must have competed representing the United States"}}
}}
```

---

## Question

{question}

Output:
"""


prompt_obj_ledger_update = """You are an Objective Evidence Ledger Annotator. Your ONLY job is to update `null` `obj` and `obj_evidence` values for each Candidate × constraint using ONLY the Search Results.

## Core Principle
- OBJ must reflect what has been objectively found in Search Results, not what the agent thinks.

## Definitions

**Candidate**
- If any entity proposed as a possible answer in the Search Results or Next Thinking, you must consider it as a Candidate.
- **If multiple names refer to the same real-world entity, you must treat them as one Candidate and aggregate.**
- Make sure to include all possible candidates uniquely in the ledger, not missing any or aggregating them unnecessarily.
- Do not assume any phrase in the question or constraints as candidates. 

**obj (Objective)**
- `true` IF:
  - Search Results explicitly and unambiguously prove the exact claim made by the constraint; OR
  - The claim can be clearly inferred from Search Results without ambiguity AND matches scope/subject/meaning.
- `false` IF:
  - Contradiction: Search Results explicitly deny the claim; OR
  - Scope mismatch: Search Results provide evidence about a related but different fact and clearly do NOT satisfy the constraint; OR
  - Exhaustion (optional): a targeted query fails to produce proof. (See "Exhaustion rule" below.)
- `null` IF:
  - No relevant Search Results evidence has been observed yet

**Evidence**
- If `obj` is `true` or `false`, `obj_evidence` MUST be a verbatim snippet from Search Results.
- If `obj` is `null`, `obj_evidence` MUST be null.

### Exhaustion rule
Only mark `obj=false` by exhaustion when BOTH are true:
1) The Search Query is clearly targeted at verifying the constraint (keywords or direct constraint phrase).
2) The Search Results contain no proof for that constraint for that Candidate.

If you cannot provide a verbatim Search Results snippet that justifies `false`, do NOT use exhaustion; keep `obj=null`.

## Rules
1. **Update only `obj` and `obj_evidence`, using ONLY verbatim information from Search Results.
   - Update `null` values, if search results provide `true` or `false` evidence for the constraint.
   - Update `true` or `false` values, only if search results provide evidence that contradicts the current values.
   - If search results doesn't provide any relevant evidence, please remain the values as they were given in the current ledger.
2. **Do NOT update `per`, `per_evidence`, or `status` in this task.**
3. Candidates start with all constraints `obj=null` unless proven otherwise.
4. Aggregate duplicate mentions of same Candidate.
5. Assess each constraint independently.

--- 

## Output Format
```json
{{
  "<Candidate Name>": {{
    "status": "active|stored|rejected", 
    "constraints": {{
      "<constraint_id>": {{
        "obj": true|false|null,
        "per": <null if new candidate, do not update otherwise>,
        "obj_evidence": "<verbatim snippet or null>",
        "per_evidence": <null if new candidate, do not update otherwise>
      }}
    }}
  }}
}}
```

## Example

### Question
Name a European country with a population above 50 million that does not use the euro and was the site of the Glorious Revolution.

### Constraints
- C1: Located in Europe
- C2: Population over 50 million
- C3: Currency other than Euro
- C4: was the site of the Glorious Revolution

### Current Ledger
{{}}

### Trajectory Segment
**Previous Thinking:**
I need to find a European country that meets all these criteria.

**Search Query:**
European countries population over 50 million non-Euro currency

**Search Results:**
The United Kingdom has a population of approximately 67 million people. It is located in Western Europe and uses the British Pound Sterling (GBP) as its currency.

**Next Thinking:**
The answer is in Europe, has 67 million people, uses the Pound. I'll answer UK.

### Output
```json
{{
  "United Kingdom": {{
    "status": "active",
    "constraints": {{
      "C1": {{"obj": true, "per": null, "obj_evidence": "located in Western Europe", "per_evidence": null}},
      "C2": {{"obj": true, "per": null, "obj_evidence": "population of approximately 67 million people", "per_evidence": null}},
      "C3": {{"obj": true, "per": null, "obj_evidence": "uses the British Pound Sterling (GBP)", "per_evidence": null}},
      "C4": {{"obj": null, "per": null, "obj_evidence": null, "per_evidence": null}}
    }}
  }}
}}
```

---

## Input

### Question
{question}

### Constraints
{constraints}

### Current Perception & Status Ledger 
{current_ledger}

### Trajectory Segment
**Previous Thinking:**
{prev_thinking}

**Search Query:**
{query}

**Search Results:** 
{result}

**Next Thinking:**
{next_thinking}

## Output
"""


prompt_per_ledger_update = """You are a Perception & Status Ledger Annotator. Your ONLY job is to update:
- candidate `status` (active|stored|rejected)
- each constraint's `per` and `per_evidence`

You MUST capture what the agent BELIEVES in the thinking text, even if it is wrong or unsupported.

## Core Principle
- PER must be derived ONLY from the agent's thinking (Previous Thinking + Next Thinking).
- Do NOT use Search Results content as evidence for per; only use it to notice candidate names.

## Definitions

**Candidate**
- If any entity proposed as a possible answer in Next Thinking, you must consider it as a Candidate.
- **If multiple names refer to the same real-world entity, you must treat them as one Candidate and aggregate.**
- Make sure to include all possible candidates uniquely in the ledger, not missing any or aggregating them unnecessarily.
- Do not assume any phrase in the question or constraints as candidates.

**Status**
- `active`: the agent is currently focusing on/selecting this candidate.
- `stored`: mentioned but not currently selected; still possible.
- `rejected`: agent explicitly rules it out or abandons it as failing.

**per (Perceived)**
- `true`: agent's thinking acts as if/assumes the constraint is satisfied.
- `false`: agent's thinking disqualifies the candidate or identifies a failure on that constraint.
- `null`: agent has not expressed a belief about that constraint.

**Evidence**
- If `per` is `true` or `false`, `per_evidence` MUST be a verbatim snippet from Next Thinking.
- If `per` is `null`, `per_evidence` MUST be null.

## Rules
1. Keep `obj` and `obj_evidence` as they were given in the current ledger, do not update them in this task.
2. Update `per` and `per_evidence` based on the agent's thinking in Next Thinking.
   - Update `null` values into `true` if agent's thinking acts as if/assumes the constraint is satisfied 
   - Update `null` values into `false` if agent's thinking disqualifies the candidate or identifies a failure on that constraint.
   - Update `true` or `false` values, only if agent's thinking states otherwise; otherwise, keep the values as they were given in the current ledger.
3. Add any new Candidates appearing in Next Thinking. New candidates start all constraints as `per=null`.
4. Status is decided from the thinking text:
   - If the agent says/acts like it will answer with X -> X is active.
   - If the agent moves away from X but doesn't rule it out -> X becomes stored.
   - If the agent says X doesn't meet constraints / is wrong -> X becomes rejected.

--- 

## Output Format
```json
{{
  "<Candidate Name>": {{
    "status": "active|stored|rejected",
    "constraints": {{
      "<constraint_id>": {{
        "obj": do not update this field,
        "per": true|false|null,
        "obj_evidence": do not update this field,
        "per_evidence": "<verbatim snippet or null>"
      }}
    }}
  }}
}}
```

## Example

### Question
Name a European country with a population above 50 million that does not use the euro and was the site of the Glorious Revolution.

### Constraints
- C1: Located in Europe
- C2: Population over 50 million
- C3: Currency other than Euro
- C4: was the site of the Glorious Revolution

### Current Ledger
{{
  "United Kingdom": {{
    "status": "active",
    "constraints": {{
      "C1": {{"obj": true, "obj_evidence": "located in Western Europe"}},
      "C2": {{"obj": true, "obj_evidence": "population of approximately 67 million people"}},
      "C3": {{"obj": true, "obj_evidence": "uses the British Pound Sterling (GBP)"}},
      "C4": {{"obj": null, "obj_evidence": null}}
    }}
  }}
}}

### Trajectory Segment
**Previous Thinking:**
I need to find a European country that meets all these criteria.

**Search Query:**
European countries population over 50 million non-Euro currency

**Search Results:**
The United Kingdom has a population of approximately 67 million people. It is located in Western Europe and uses the British Pound Sterling (GBP) as its currency.

**Next Thinking:**
The answer is in Europe, has 67 million people, uses the Pound. I'll answer UK.

### Output
```json
{{
  "United Kingdom": {{
    "status": "active",
    "constraints": {{
      "C1": {{"obj": true, "per": true, "obj_evidence": "located in Western Europe", "per_evidence": "it's in Europe"}},
      "C2": {{"obj": true, "per": true, "obj_evidence": "population of approximately 67 million people", "per_evidence": "has 67 million people"}},
      "C3": {{"obj": true, "per": true, "obj_evidence": "uses the British Pound Sterling (GBP)", "per_evidence": "uses the Pound"}},
      "C4": {{"obj": null, "per": null, "obj_evidence": null, "per_evidence": null}}
    }}
  }}
}}
```

---

## Input

### Question
{question}

### Constraints
{constraints}

### Current Perception & Status Ledger 
{current_ledger}

### Trajectory Segment
**Previous Thinking:**
{prev_thinking}

**Search Query:**
{query}

**Search Results:** 
{result}

**Next Thinking:**
{next_thinking}

## Output
"""
