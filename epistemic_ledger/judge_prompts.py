"""Answer-correctness prompts (BioASQ has its own)."""

PROMPT = """
### Instruction
You are an impartial evaluator.

You will be given:
- a **question**
- a **gold (reference) answer**, which may contain **one or more valid entities**
- a **predicted answer**

Your task is to determine whether the **predicted answer is correct**.

### Rules (follow strictly)
1. If the gold answer contains **multiple valid entities**, the predicted answer(s) is **correct if and only if** it clearly matches **any one** of the gold entities.
2. If the gold answer contains a **single entity**, the predicted answer(s) must match **that exact entity**.
3. Minor surface differences (e.g., capitalization, abbreviations, aliases, name order) are allowed **only if** they unambiguously refer to the same entity.
4. If the predicted answer(s) refers to a **different entity**, the verdict must be **false**, even if it partially satisfies the question.
5. If the predicted answer(s) is **more general, more specific, or a different category** than the gold answer(s), the verdict must be **false**.
6. Do **not** use outside knowledge beyond comparing the gold and predicted answers.
7. If equivalence or membership is **ambiguous**, default to **false**.
8. Do not reward partial correctness.

### Output Format (exactly)
```json
{{
  "verdict": true or false,
  "justification": "One-sentence explanation."
}}

### Inputs
- question: {question}
- gold answer: {answer}
- predicted answer: {predicted_answer}
"""

BIOASQ_PROMPT = """
### Instruction
You are an impartial evaluator for the BioASQ biomedical QA benchmark.

You will be given:
- a **question**
- a **gold (reference) answer**
- a **predicted answer**

Your task is to determine whether the **predicted answer is correct**.

BioASQ answers fall into one of these forms:
- **Yes/No**: the gold begins with "Yes" or "No".
- **Factoid (single entity)**: the gold identifies one canonical entity (a tool, gene, drug, protein, disease, etc.).
- **List**: the gold enumerates multiple required items (e.g., "Nanog, Pou5f1 and SoxB1"; "MET, CMT, and DRM"; "Measles, mumps and encephalitis").

### Rules (follow strictly)
1. First identify which form the gold answer takes and what the **required items** are.
2. **List answers**: the predicted answer is correct **if and only if it explicitly names every required item** from the gold list. Naming only a proper subset is **incorrect**, even if those items are right. Extra items are tolerated only if they don't contradict the gold.
3. **Factoid answers**: the predicted answer must match **the single required entity exactly**. Aliases, abbreviations, and case/format differences are accepted **only when they unambiguously refer to the same entity**.
4. **Yes/No answers**: the predicted answer must agree on the yes/no verdict. If the gold also enumerates supporting items, rule 2 additionally applies to those items.
5. If the predicted answer is **more general, more specific, or a different category** than the gold (e.g., a parent class, an unrelated tool that does something similar), the verdict is **false**.
6. Do **not** use outside biomedical knowledge beyond comparing the gold and predicted text.
7. If membership or equivalence is **ambiguous**, default to **false**.
8. Do **not** reward partial correctness.

### Output Format (exactly)
```json
{{
  "verdict": true or false,
  "justification": "One-sentence explanation that names the required items and which (if any) are missing from the prediction."
}}

### Inputs
- question: {question}
- gold answer: {answer}
- predicted answer: {predicted_answer}
"""
