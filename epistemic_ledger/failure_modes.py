import json
import argparse
import os
from glob import glob


# =============================================================================
# Constants
# =============================================================================

KNOWN_BASELINES = [
    "search-r1",
    "asearcher",
    "rag-r1",
    "dr-tulu",
    "webexplorer",
    "search_o1_gpt-oss-20b",
    "search_o1_gpt-oss-120b",
    "tongyidr",
    "tongyidr-liveledger-4B",
    "react_20B",
    "react_20B_liveledger_4B",
    "react",
    "react_s1",
    "react_120B_liveledger_4B",
]
RUN_SUFFIXES = ["", "_2", "_3"]
DATASET_CHOICES = ["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa", "bioasq"]

FAILURE_MODES = ["Bare Assertion", "Overlooked Refutation", "Stagnation", "Premature Exit", "None"]

# Cap bioasq evaluation to the first N items (by item index) to keep its
# weight comparable to the other datasets.


# =============================================================================
# Argument Parser
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger_dir", type=str, default="epistemic_ledger",
        help="Directory containing ledger runs. Each baseline has subdirs "
             "{baseline}, {baseline}_2, {baseline}_3 for the three runs."
    )
    parser.add_argument(
        "--run_suffixes", nargs="+", type=str, default=RUN_SUFFIXES,
        help="Suffixes appended to each baseline name to locate the three runs."
    )
    parser.add_argument(
        "--baseline_name", "-b",
        nargs="+", type=str,
        default=KNOWN_BASELINES,
        help=f"Base baseline names (without run suffix). Known: {KNOWN_BASELINES}."
    )
    parser.add_argument(
        "--dataset_name", "-d",
        nargs="+", type=str,
        default=DATASET_CHOICES
    )
    return parser.parse_args()


# =============================================================================
# Evaluator Class
# =============================================================================

class FailureModeEvaluator:
    """Evaluates failure modes based on ledger analysis."""

    def __init__(self, dataset_name, baseline_name):
        self.dataset_name = dataset_name
        self.baseline_name = baseline_name
        self.failure_modes = {mode: 0 for mode in FAILURE_MODES}

    def evaluate(self, ledgers, checklist, finished):
        """Main evaluation entry point."""
        # Clean ledgers
        for ledger in ledgers:
            if "" in ledger:
                del ledger[""]

        # Handle empty ledgers case
        if len(ledgers) == 0:
            self.failure_modes["Premature Exit"] += 1
            return self.failure_modes

        # Evaluate each failure mode
        none = self.evaluate_none(ledgers, finished)
        bare_assertion = self.evaluate_bare_assertion(ledgers)
        overlooked_refutation = self.evaluate_overlooked_refutation(ledgers)
        stagnation = self.evaluate_stagnation(ledgers)
        premature_exit = self.evaluate_premature_exit(ledgers)

        # Case 1: Sound (no failure mode)
        if none:
            self.failure_modes["None"] += 1
            return self.failure_modes

        # Case 2: Unsound (one or more failure modes)
        if bare_assertion:
            self.failure_modes["Bare Assertion"] += 1
        if overlooked_refutation:
            self.failure_modes["Overlooked Refutation"] += 1
        if stagnation:
            self.failure_modes["Stagnation"] += 1
        if premature_exit:
            self.failure_modes["Premature Exit"] += 1

        if not (none or bare_assertion or overlooked_refutation or stagnation or premature_exit):
            raise ValueError("No failure mode found")

        return self.failure_modes

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_final_ledger(self, ledgers):
        """Get the final valid ledger (dict type)."""
        final_ledger = ledgers[-1]
        if type(final_ledger) != dict:
            if len(ledgers) > 1:
                final_ledger = ledgers[-2]
            else:
                return None
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

    def evaluate_bare_assertion(self, ledgers):
        """Check for bare assertion: obj=None, per=True with evidence."""
        final_ledger = self._get_final_ledger(ledgers)
        if final_ledger is None:
            return False

        for cand_name, data in final_ledger.items():
            if data['status'] == 'active':
                for c in data['constraints'].values():
                    if c['obj'] is None and c['per'] is True and c['per_evidence'] is not None:
                        return True
        return False

    def evaluate_overlooked_refutation(self, ledgers):
        """Check for overlooked refutation: obj=False with evidence but still active."""
        final_ledger = self._get_final_ledger(ledgers)
        if final_ledger is None:
            return False

        for cand_name, data in final_ledger.items():
            if data['status'] == 'active':
                for c in data['constraints'].values():
                    if c['obj'] is False and c.get('obj_evidence', '') is not None:
                        return True
        return False

    def evaluate_stagnation(self, ledgers):
        """Check for stagnation: no progress in last 3 turns."""
        final_ledger = self._get_final_ledger(ledgers)
        if final_ledger is None:
            return False

        final_cand = self._get_active_candidates(final_ledger)

        # Check if any candidate is fully verified
        for cand_name, data in final_cand.items():
            if self.is_all_true(data):
                return False

        # Ignore candidates whose obj is all null (not yet investigated)
        meaningful_cand = {
            k: v for k, v in final_cand.items()
            if any(c['obj'] is not None for c in v['constraints'].values())
        }

        # Check if no progress for more than 3 turns
        if len(meaningful_cand) == 0:
            for ledger in reversed(ledgers[-3:-1]):
                if len(self._get_active_candidates(ledger)) > 0:
                    return False
        else:
            for ledger in reversed(ledgers[-3:-1]):
                curr_cand = self._get_active_candidates(ledger)
                for cand_name, data in meaningful_cand.items():
                    if cand_name not in curr_cand:
                        return False
                    if not self.has_same_ledger(data['constraints'], curr_cand[cand_name]['constraints']):
                        return False
        return True

    def evaluate_premature_exit(self, ledgers):
        """Check for premature exit: unverified constraints or no active candidates."""
        final_ledger = self._get_final_ledger(ledgers)
        if final_ledger is None:
            return True

        for cand_name, data in final_ledger.items():
            if data['status'] == 'active':
                for c in data['constraints'].values():
                    if c['obj'] is None and c['per'] is not True:
                        return True

        if all(c['status'] != 'active' for c in final_ledger.values()):
            return True

        return False

    def evaluate_none(self, ledgers, finished):
        """Check if verification is complete: active candidate with all obj=True."""
        final_ledger = self._get_final_ledger(ledgers)
        if final_ledger is None:
            return False

        for _, data in final_ledger.items():
            if data['status'] == 'active':
                if all(c['obj'] for c in data['constraints'].values()):
                    return True
        return False


# =============================================================================
# Mean / Std across runs
# =============================================================================

def compute_mean_std(per_run_totals, per_run_n, metric):
    """Compute mean and std of (totals[metric] / n) across runs that have n>0."""
    values = []
    for suffix, totals in per_run_totals.items():
        n = per_run_n.get(suffix, 0)
        if n > 0:
            values.append(totals[metric] / n)
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) > 1:
        # sample std (ddof=1)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = var ** 0.5
    else:
        std = None
    return mean, std


def fmt_mean_std(mean, std):
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{std:.1f}"


# =============================================================================
# Result Printing
# =============================================================================

def print_results_table(baselines, per_run_totals, per_run_n, columns):
    """Print a formatted results table with mean±std across runs."""
    baseline_col_width = max(len(b) for b in baselines) + 2
    # widen column to fit "xx.x±yy.y"
    col_width = max(max(len(c) for c in columns), 10) + 2

    header = "Baseline".ljust(baseline_col_width) + "".join(c.rjust(col_width) for c in columns)
    print(header)
    print("-" * len(header))

    for baseline_name in baselines:
        row = baseline_name.ljust(baseline_col_width)
        totals = per_run_totals[baseline_name]
        ns = per_run_n[baseline_name]

        if any(n > 0 for n in ns.values()):
            for c in columns:
                mean, std = compute_mean_std(totals, ns, c)
                row += fmt_mean_std(mean, std).rjust(col_width)
        else:
            row += "".join("0".rjust(col_width) for _ in columns)
        print(row)


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Per-run accumulators across all datasets
    # overall_per_run[baseline][suffix][mode] = sum over datasets of (count * 100)
    # overall_per_run_n[baseline][suffix] = sum over datasets of #items
    overall_per_run = {
        b: {s: {m: 0 for m in FAILURE_MODES} for s in args.run_suffixes}
        for b in args.baseline_name
    }
    overall_per_run_n = {
        b: {s: 0 for s in args.run_suffixes}
        for b in args.baseline_name
    }

    columns = FAILURE_MODES

    # Per-dataset snapshots kept around for cache export.
    per_dataset_snapshots = {}

    # Process each dataset
    for dataset_name in args.dataset_name:
        print("=" * 31)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 31)

        per_dataset_per_run = {
            b: {s: {m: 0 for m in FAILURE_MODES} for s in args.run_suffixes}
            for b in args.baseline_name
        }
        per_dataset_per_run_n = {
            b: {s: 0 for s in args.run_suffixes}
            for b in args.baseline_name
        }

        for baseline_name in args.baseline_name:
            # Iterate independently per run, so each run produces its own taxonomy.
            for suffix in args.run_suffixes:
                run_baseline = f"{baseline_name}{suffix}"
                run_dir = os.path.join(args.ledger_dir, run_baseline, dataset_name)
                if not os.path.isdir(run_dir):
                    continue

                output_files = glob(os.path.join(run_dir, "*.json"))
                output_files = [fn for fn in output_files if not fn.endswith("_.json")]
                output_files = sorted(
                    output_files,
                    key=lambda x: int(x.split("/")[-1].split("_")[-1].replace(".json", ""))
                )
                for ledger_file in output_files:
                    with open(ledger_file, "r") as f:
                        ledger_data = json.load(f)
                        ledgers = ledger_data["ledger"]
                        checklist = ledger_data["checklist"]["checklist"]
                        finished = ledger_data.get("finished", True)

                    empty_content = ledger_data.get("content", " ") == ""

                    evaluator = FailureModeEvaluator(dataset_name, baseline_name)
                    try:
                        failure_modes = evaluator.evaluate(ledgers, checklist, finished)
                    except Exception as e:
                        print(f"Error evaluating {run_baseline} {dataset_name} {ledger_file}: {e}")
                        continue

                    if empty_content:
                        failure_modes["Premature Exit"] = 1
                        failure_modes["None"] = 0

                    for mode, count in failure_modes.items():
                        per_dataset_per_run[baseline_name][suffix][mode] += 100 if count > 0 else 0
                    per_dataset_per_run_n[baseline_name][suffix] += 1

            # Accumulate to overall
            for suffix in args.run_suffixes:
                for mode, val in per_dataset_per_run[baseline_name][suffix].items():
                    overall_per_run[baseline_name][suffix][mode] += val
                overall_per_run_n[baseline_name][suffix] += per_dataset_per_run_n[baseline_name][suffix]

        # Print per-dataset results
        print_results_table(
            args.baseline_name,
            per_dataset_per_run,
            per_dataset_per_run_n,
            columns
        )
        print()

        # Snapshot for cache.
        per_dataset_snapshots[dataset_name] = {
            "per_run_totals": per_dataset_per_run,
            "per_run_n": per_dataset_per_run_n,
        }

    print_results_table(
        args.baseline_name,
        overall_per_run,
        overall_per_run_n,
        columns
    )
    print()

    print(overall_per_run_n)

if __name__ == "__main__":
    args = get_args()
    main(args)
