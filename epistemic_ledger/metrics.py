"""Acc, UAR and constraint outcomes from judged Epistemic Ledgers (the numbers in the paper).

Paper settings, always on: evidence is carried forward across turns (ledger_merge.merge_obj), and an answer forced after
the turn cap counts as no answer. A question without an answer is incorrect and unsubstantiated.

  Acc        % of questions answered correctly (is_correct from judge.py)
  UAR        unsubstantiated answer rate: % of questions whose final answer has a constraint that retrieved evidence
             does not establish (plus every question without an answer)
  Verified / Assumed / Refuted / Unchecked
             % of the committed candidate's constraints, averaged over questions:
             Verified  evidence establishes it          Assumed    no evidence, but the agent states it holds
             Refuted   evidence contradicts it           Unchecked  no evidence, and the agent never addresses it
  C-V / C-UV / IC-V / IC-UV   correct or incorrect x verified or unsubstantiated

    python epistemic_ledger/metrics.py --ledger_dir ledgers [-a agent ...] [--per_dataset] [--out_csv results.csv]
"""
import argparse
import collections
import csv
import json
import os
from glob import glob

from ledger_merge import merge_obj

DATASET_CHOICES = ["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa", "bioasq"]
ACCURACY_METRICS = ["Correct", "Incorrect", "Verified", "Underverified",
                    "C - V", "C - UV", "IC - V", "IC - UV", "Turns"]
OUTCOMES = ["Verified", "Assumed", "Refuted", "Unchecked"]


class LocalMinimaAccuracyEvaluator:
    """Correct/Incorrect x Verified/Underverified (= unsubstantiated) of one trajectory."""

    def __init__(self, dataset_name, baseline_name):
        self.dataset_name = dataset_name
        self.baseline_name = baseline_name
        self.local_minima_accuracy = {metric: False for metric in ACCURACY_METRICS}

    def evaluate(self, ledgers, checklist, is_correct):
        """Main evaluation entry point."""
        # Clean ledgers
        for ledger in ledgers:
            if "" in ledger:
                del ledger[""]

        # Handle empty ledgers case
        if len(ledgers) == 0:
            self.local_minima_accuracy["Underverified"] = True
            if is_correct:
                self.local_minima_accuracy["Correct"] = True
                self.local_minima_accuracy["C - UV"] = True
            else:
                self.local_minima_accuracy["Incorrect"] = True
                self.local_minima_accuracy["IC - UV"] = True
            return self.local_minima_accuracy

        # Evaluate each failure mode
        none = self.evaluate_none(ledgers)
        ungrounded_assumption = self.evaluate_ungrounded_assumption(ledgers)
        delusion = self.evaluate_delusion(ledgers)
        stagnation = self.evaluate_stagnation(ledgers)
        premature_exit = self.evaluate_premature_exit(ledgers)

        # Set correctness
        if is_correct:
            self.local_minima_accuracy["Correct"] = True
        else:
            self.local_minima_accuracy["Incorrect"] = True

        # Case 1: Verified (no failure mode)
        if none:
            self.local_minima_accuracy["Verified"] = True
            if is_correct:
                self.local_minima_accuracy["C - V"] = True
            else:
                self.local_minima_accuracy["IC - V"] = True
            return self.local_minima_accuracy

        # Case 2: Underverified (any failure mode present)
        if ungrounded_assumption or delusion or stagnation or premature_exit:
            self.local_minima_accuracy["Underverified"] = True
            if is_correct:
                self.local_minima_accuracy["C - UV"] = True
            else:
                self.local_minima_accuracy["IC - UV"] = True
            return self.local_minima_accuracy

        # Should not reach here
        print(json.dumps(ledgers[-1], indent=4))
        raise ValueError("No failure mode found")

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_final_ledger(self, ledgers):
        """Get the final valid ledger (dict type)."""
        final_ledger = ledgers[-1]
        if type(final_ledger) != dict:
            final_ledger = ledgers[-2]
        return final_ledger

    def _get_active_candidates(self, ledger):
        """Get active candidates from a ledger."""
        try:
            return {k: v for k, v in ledger.items() if v['status'] == 'active'}
        except:
            return {}

    def is_all_true(self, data):
        """Check if all constraints have obj=True."""
        for c in data['constraints'].values():
            if c['obj'] is not True:
                return False
        return True

    def has_same_ledger(self, cand1_data, cand2_data):
        """Check if two candidate ledgers are identical."""
        for constraint_name, constraint_data in cand1_data.items():
            if constraint_name not in cand2_data:
                return False
            if constraint_data['obj'] != cand2_data[constraint_name]['obj']:
                return False
            if constraint_data['per'] != cand2_data[constraint_name]['per']:
                return False
        return True

    # -------------------------------------------------------------------------
    # Failure Mode Evaluators
    # -------------------------------------------------------------------------

    def evaluate_ungrounded_assumption(self, ledgers):
        """Check for ungrounded assumption: obj=None, per=True with evidence."""
        final_ledger = self._get_final_ledger(ledgers)

        for cand_name, data in final_ledger.items():
            if data['status'] == 'active':
                for c in data['constraints'].values():
                    if c['obj'] is None and c['per'] is True and c['per_evidence'] is not None:
                        return True
        return False

    def evaluate_delusion(self, ledgers):
        """Check for delusion: obj=False with evidence but still active."""
        final_ledger = self._get_final_ledger(ledgers)

        for cand_name, data in final_ledger.items():
            if data['status'] == 'active':
                for c in data['constraints'].values():
                    if c['obj'] is False and c.get('obj_evidence', '') is not None:
                        return True
        return False

    def evaluate_stagnation(self, ledgers):
        """Check for stagnation: no progress in last 3 turns."""
        final_ledger = self._get_final_ledger(ledgers)
        final_cand = self._get_active_candidates(final_ledger)

        # Check if any candidate is fully verified
        for cand_name, data in final_cand.items():
            if self.is_all_true(data):
                return False

        # Check if no progress for more than 3 turns
        if len(final_cand) == 0:
            for ledger in reversed(ledgers[-3:-1]):
                if len(self._get_active_candidates(ledger)) > 0:
                    return False
        else:
            for ledger in reversed(ledgers[-3:-1]):
                curr_cand = self._get_active_candidates(ledger)
                for cand_name, data in final_cand.items():
                    if cand_name not in ledger:
                        return False
                for cand_name, data in ledger.items():
                    if data['status'] == 'active' and cand_name in curr_cand:
                        if not self.has_same_ledger(data['constraints'], curr_cand[cand_name]['constraints']):
                            return False
        return True

    def evaluate_premature_exit(self, ledgers):
        """Check for premature exit: unverified constraints or no active candidates."""
        final_ledger = self._get_final_ledger(ledgers)

        for cand_name, data in final_ledger.items():
            if data['status'] == 'active':
                for c in data['constraints'].values():
                    if c['obj'] is None and c['per'] is not True:
                        return True

        if all(c['status'] != 'active' for c in final_ledger.values()):
            return True

        return False

    def evaluate_none(self, ledgers):
        """Check if verification is complete: active candidate with all obj=True."""
        final_ledger = self._get_final_ledger(ledgers)

        for _, data in final_ledger.items():
            if data['status'] == 'active':
                if all(c['obj'] for c in data['constraints'].values()):
                    return True
        return False


# =============================================================================
# Constraint outcomes of the committed candidate
# =============================================================================

def constraint_label(c):
    """Verified / Assumed / Refuted / Unchecked of one constraint; None = the agent states it does NOT hold (excluded)."""
    o, p = c.get("obj"), c.get("per")
    if p is False:
        return None
    return "Verified" if o is True else "Refuted" if o is False else "Assumed" if p is True else "Unchecked"


def constraint_outcomes(d):
    """% of the committed candidate's constraints per outcome. Committed candidate = an active candidate in the final
    ledger with every constraint verified, else the active one with most verified constraints. No tracked candidate =
    100 % Unchecked."""
    L = merge_obj(d)
    final = L[-1] if L else {}
    act = {k: v for k, v in final.items() if isinstance(v, dict) and v.get("status") == "active" and v.get("constraints")}
    out = {k: 0.0 for k in OUTCOMES}
    if act:
        full = [k for k, v in act.items() if all(x.get("obj") is True for x in v["constraints"].values())]
        a = full[0] if full else max(act, key=lambda k: sum(x.get("obj") is True for x in act[k]["constraints"].values()))
        labels = [constraint_label(x) for x in act[a]["constraints"].values()]
        labels = [x for x in labels if x is not None]
        if labels:
            for x in labels:
                out[x] += 100 / len(labels)
            return out
    out["Unchecked"] = 100.0
    return out


# =============================================================================
# Table
# =============================================================================

ACC_COLUMNS = [("Acc", "Correct"), ("UAR", "Underverified"),
               ("C-V", "C - V"), ("C-UV", "C - UV"), ("IC-V", "IC - V"), ("IC-UV", "IC - UV")]


def score_agent(ledger_dir, agent, datasets):
    acc_tot, n_acc = collections.Counter(), 0
    out_tot, n_out = collections.Counter(), 0
    for ds in datasets:
        for p in sorted(glob(os.path.join(ledger_dir, agent, ds, "item_*.json"))):
            d = json.load(open(p))
            for k, v in constraint_outcomes(d).items():
                out_tot[k] += v
            n_out += 1
            if "forced" in str(d.get("termination") or ""):   # an answer forced after the turn cap = no answer
                d["content"] = ""
            empty_content = d.get("content", " ") == ""
            try:
                lab = LocalMinimaAccuracyEvaluator(ds, agent).evaluate(merge_obj(d), d["checklist"]["checklist"],
                                                                       d.get("is_correct", False))
            except Exception as e:  # noqa: BLE001  (the original code skips such items)
                print(f"skipped {p}: {e}")
                continue
            if empty_content:
                lab.update({"Correct": False, "Incorrect": True, "Verified": False, "Underverified": True,
                            "IC - UV": True, "C - UV": False, "C - V": False, "IC - V": False})
            n_acc += 1
            for k, v in lab.items():
                acc_tot[k] += 1 if v else 0
    row = {"agent": agent, "n": n_acc}
    for col, key in ACC_COLUMNS:
        row[col] = round(100 * acc_tot[key] / n_acc, 1) if n_acc else None
    for k in OUTCOMES:
        row[k] = round(out_tot[k] / n_out, 1) if n_out else None
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger_dir", required=True)
    ap.add_argument("--agents", "-a", nargs="*", help="default: every folder in --ledger_dir")
    ap.add_argument("--datasets", "-d", nargs="+", default=DATASET_CHOICES)
    ap.add_argument("--per_dataset", action="store_true", help="also print one table per dataset")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()
    agents = a.agents or sorted(x for x in os.listdir(a.ledger_dir) if os.path.isdir(os.path.join(a.ledger_dir, x)))
    views = [("all", a.datasets)] + ([(ds, [ds]) for ds in a.datasets] if a.per_dataset else [])
    cols = ["Acc", "UAR"] + OUTCOMES + [c for c, _ in ACC_COLUMNS[2:]]
    rows = []
    for view, dss in views:
        print(f"\n## {view}\n\n| Agent | n | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
        for ag in agents:
            r = score_agent(a.ledger_dir, ag, dss)
            print(f"| {r['agent']} | {r['n']} | " + " | ".join("-" if r[c] is None else f"{r[c]:.1f}" for c in cols) + " |")
            rows.append({"view": view, **r})
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
