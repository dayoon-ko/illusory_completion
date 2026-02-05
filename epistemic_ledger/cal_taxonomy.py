import json
import argparse
import os
from glob import glob


# =============================================================================
# Constants
# =============================================================================

# Known baselines (for reference, any baseline name is accepted)
KNOWN_BASELINES = [
    "search-r1", "hds", "rag-r1", "asearcher", "dr-tulu", "webexplorer",
    "tongyidr", "tongyidr-liveledger-20b", 
    "search_o1_gpt-oss-20b", "search_o1_gpt-oss-120b",
    "react_20b", "react_liveledger_20b",
    "react", "react_s1", "react_liveledger",
]

DATASET_CHOICES = ["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa"]

FAILURE_MODES = ["Bare Assertion", "Overlooked Refutation", "Stagnation", "Premature Exit", "None"]


# =============================================================================
# Argument Parser
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger_dir", type=str, default="output/epistemic_ledger_finished")
    parser.add_argument(
        "--baseline_name", "-b",
        nargs="+", type=str,
        default=[
            "search-r1", "hds", "rag-r1", "asearcher", "dr-tulu", "webexplorer",
            "tongyidr", "tongyidr-liveledger-20b", 
            "search_o1_gpt-oss-20b", "search_o1_gpt-oss-120b",
            "react_20b", "react_liveledger_20b",
            "react", "react_s1", "react_liveledger"
        ],
        help=f"Baseline names. Known: {KNOWN_BASELINES}. Any baseline is supported."
    )
    parser.add_argument(
        "--dataset_name", "-d",
        nargs="+", type=str,
        default=DATASET_CHOICES
    )
    parser.add_argument(
        "--output_mode", "-o",
        type=str, default="all",
        choices=["count", "global_ratio", "local_ratio", "all"],
        help="Output mode: 'count' for raw counts, 'global_ratio' for ratio over total datapoints, 'local_ratio' for ratio over failure datapoints, 'all' for all"
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
# Result Printing
# =============================================================================

def print_counts_table(baselines, results, columns, baseline_col_width, col_width):
    """Print raw counts table."""
    print("\n[Raw Counts]")
    header = "Baseline".ljust(baseline_col_width) + "".join(c.rjust(col_width) for c in columns)
    print(header)
    print("-" * len(header))
    
    for baseline_name in baselines:
        row = baseline_name.ljust(baseline_col_width)
        for c in columns:
            row += str(results[baseline_name][c]).rjust(col_width)
        print(row)


def print_global_ratio_table(baselines, results, total_items, columns, baseline_col_width, col_width):
    """Print global ratio table (over total datapoints)."""
    print("\n[Global Ratio - over Total Datapoints]")
    header = "Baseline".ljust(baseline_col_width) + "".join(c.rjust(col_width) for c in columns)
    print(header)
    print("-" * len(header))
    
    for baseline_name in baselines:
        row = baseline_name.ljust(baseline_col_width)
        total = total_items[baseline_name] if isinstance(total_items[baseline_name], int) else sum(total_items[baseline_name].values())
        for c in columns[:-1]:
            if total > 0:
                row += str(round(results[baseline_name][c] / total, 2)).rjust(col_width)
            else:
                row += "0".rjust(col_width)
        print(row)


def print_local_ratio_table(baselines, results, total_items, columns, baseline_col_width, col_width):
    """Print local ratio table (over failure datapoints)."""
    print("\n[Local Ratio - over Failure Datapoints]")
    header = "Baseline".ljust(baseline_col_width) + "".join(c.rjust(col_width) for c in columns)
    print(header)
    print("-" * len(header))
    
    for baseline_name in baselines:
        row = baseline_name.ljust(baseline_col_width)
        total = total_items[baseline_name] if isinstance(total_items[baseline_name], int) else sum(total_items[baseline_name].values())
        failure_count = total - results[baseline_name]["None"]
        for c in columns[:-1]:
            if failure_count > 0:
                row += str(round(results[baseline_name][c] / failure_count, 2)).rjust(col_width)
            else:
                row += "0".rjust(col_width)
        print(row)


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Initialize accumulators
    overall_failure_modes = {
        baseline_name: {mode: 0 for mode in FAILURE_MODES}
        for baseline_name in args.baseline_name
    }
    total_items_overall = {
        baseline_name: {dataset_name: 0 for dataset_name in args.dataset_name}
        for baseline_name in args.baseline_name
    }
    
    columns = FAILURE_MODES
    
    # Process each dataset
    for dataset_name in args.dataset_name:
        print("=" * 31)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 31)
        
        all_results = {}
        
        for baseline_name in args.baseline_name:
            # Get input files
            input_dir = os.path.join(args.ledger_dir, baseline_name, dataset_name)
            output_files = glob(os.path.join(input_dir, "*.json"))
            output_files = [fn for fn in output_files if not fn.endswith("_.json")]
            output_files = sorted(
                output_files,
                key=lambda x: int(x.split("/")[-1].split("_")[-1].replace(".json", ""))
            )
            
            total_failure_modes = {mode: 0 for mode in FAILURE_MODES}
            
            # Process each file
            for file in output_files:
                with open(file, "r") as f:
                    data = json.load(f)
                    ledgers = data["ledger"]
                    finished = data["finished"]
                    checklist = data["checklist"]["checklist"]
                
                # Evaluate
                evaluator = FailureModeEvaluator(dataset_name, baseline_name)
                try:
                    failure_modes = evaluator.evaluate(ledgers, checklist, finished)
                    
                    # Override for empty content
                    if data.get("content", " ") == "":
                        failure_modes["Premature Exit"] = 1
                        failure_modes["None"] = 0
                    
                    # Debug outputs
                    if all(count == 0 for count in failure_modes.values()):
                        print(file, failure_modes)
                    if failure_modes["None"] and sum(count > 0 for count in failure_modes.values()) > 1:
                        print(file, failure_modes)
                        
                except Exception as e:
                    print(f"Error evaluating {baseline_name} {dataset_name} {file}: {e}")
                
                # Accumulate results
                for mode, count in failure_modes.items():
                    total_failure_modes[mode] += 1 if count > 0 else 0
                
                # Debug: conflicting failure modes
                if any(count > 0 for mode, count in failure_modes.items() if mode != "None") and failure_modes["None"] > 0:
                    print(json.dumps(ledgers[-1], indent=2))
                    print(f"Error evaluating {baseline_name} {dataset_name} {file}")
                    print(failure_modes)
            
            all_results[baseline_name] = total_failure_modes
            
            # Accumulate to overall
            for mode, count in total_failure_modes.items():
                overall_failure_modes[baseline_name][mode] += count
            total_items_overall[baseline_name][dataset_name] += len(output_files)
        
        # Print per-dataset results
        baseline_col_width = max(len(b) for b in args.baseline_name) + 2
        col_width = max(len(c) for c in columns) + 2
        
        if args.output_mode in ["count", "all"]:
            print_counts_table(args.baseline_name, all_results, columns, baseline_col_width, col_width)
        
        if args.output_mode in ["global_ratio", "all"]:
            print_global_ratio_table(
                args.baseline_name, all_results,
                {b: total_items_overall[b][dataset_name] for b in args.baseline_name},
                columns, baseline_col_width, col_width
            )
        
        if args.output_mode in ["local_ratio", "all"]:
            print_local_ratio_table(
                args.baseline_name, all_results,
                {b: total_items_overall[b][dataset_name] for b in args.baseline_name},
                columns, baseline_col_width, col_width
            )
        print()
    
    # Print overall summary
    print("=" * 31)
    print("OVERALL (across all datasets)")
    print("=" * 31)
    
    baseline_col_width = max(len(b) for b in args.baseline_name) + 2
    col_width = max(len(c) for c in columns) + 2
    
    if args.output_mode in ["count", "all"]:
        print_counts_table(args.baseline_name, overall_failure_modes, columns, baseline_col_width, col_width)
    
    if args.output_mode in ["global_ratio", "all"]:
        print_global_ratio_table(
            args.baseline_name, overall_failure_modes, total_items_overall,
            columns, baseline_col_width, col_width
        )
    
    if args.output_mode in ["local_ratio", "all"]:
        print_local_ratio_table(
            args.baseline_name, overall_failure_modes, total_items_overall,
            columns, baseline_col_width, col_width
        )
    print()
    
    print(total_items_overall)


if __name__ == "__main__":
    args = get_args()
    main(args)
