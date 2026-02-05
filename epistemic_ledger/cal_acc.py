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
    "react", "react_tts", "react_liveledger",
]

DATASET_CHOICES = ["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa"]

ACCURACY_METRICS = ["Correct", "Incorrect", "Verified", "Underverified",
                    "C - V", "C - UV", "IC - V", "IC - UV"]


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
            "react", "react_tts", "react_liveledger"
        ],
        help=f"Baseline names. Known: {KNOWN_BASELINES}. Any baseline is supported."
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

class LocalMinimaAccuracyEvaluator:
    """Evaluates local minima accuracy based on ledger analysis."""
    
    def __init__(self, dataset_name, baseline_name):
        self.dataset_name = dataset_name
        self.baseline_name = baseline_name
        self.local_minima_accuracy = {metric: False for metric in ACCURACY_METRICS}
    
    def evaluate(self, ledgers, checklist, finished, is_correct):
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
        none = self.evaluate_none(ledgers, finished)
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

    def evaluate_none(self, ledgers, finished):
        """Check if verification is complete: active candidate with all obj=True."""
        final_ledger = self._get_final_ledger(ledgers)
        
        for _, data in final_ledger.items():
            if data['status'] == 'active':
                if all(c['obj'] for c in data['constraints'].values()):
                    return True
        return False


# =============================================================================
# Result Printing
# =============================================================================

def print_results_table(baselines, results, total_items, columns):
    """Print a formatted results table."""
    baseline_col_width = max(len(b) for b in baselines) + 2
    col_width = max(len(c) for c in columns) + 2
    
    header = "Baseline".ljust(baseline_col_width) + "".join(c.rjust(col_width) for c in columns)
    print(header)
    print("-" * len(header))
    
    for baseline_name in baselines:
        row = baseline_name.ljust(baseline_col_width)
        n = total_items[baseline_name] if isinstance(total_items[baseline_name], int) else sum(total_items[baseline_name].values())
        res = results[baseline_name]
        
        if n > 0:
            acc = res["Correct"] / n
            uar = res["Underverified"] / n
            row += str(round(acc, 1)).rjust(col_width)
            row += str(round(uar, 1)).rjust(col_width)
            for c in columns[2:]:
                row += str(round(res[c] / n, 1)).rjust(col_width)
        else:
            row += "".join("0".rjust(col_width) for _ in columns)
        print(row)


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Initialize accumulators
    overall_accuracy = {
        baseline_name: {metric: 0 for metric in ACCURACY_METRICS}
        for baseline_name in args.baseline_name
    }
    total_items_overall = {
        baseline_name: {dataset_name: 0 for dataset_name in args.dataset_name}
        for baseline_name in args.baseline_name
    }
    
    columns = ["Acc", "UAR", "C - V", "C - UV", "IC - V", "IC - UV"]
    
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
            
            total_accuracy = {metric: 0 for metric in ACCURACY_METRICS}
            
            # Process each file
            for file in output_files:
                with open(file, "r") as f:
                    data = json.load(f)
                    ledgers = data["ledger"]
                    finished = data["finished"]
                    is_correct = data["is_correct"]
                    checklist = data["checklist"]["checklist"]
                
                # Evaluate
                evaluator = LocalMinimaAccuracyEvaluator(dataset_name, baseline_name)
                try:
                    local_minima_accuracy = evaluator.evaluate(ledgers, checklist, finished, is_correct)
                    
                    # Override for empty content
                    if data.get("content", " ") == "":
                        local_minima_accuracy["Correct"] = False
                        local_minima_accuracy["Incorrect"] = True
                        local_minima_accuracy["Verified"] = False
                        local_minima_accuracy["Underverified"] = True
                        local_minima_accuracy["IC - UV"] = True
                        local_minima_accuracy["C - UV"] = False
                        local_minima_accuracy["C - V"] = False
                        local_minima_accuracy["IC - V"] = False
                        
                except Exception as e:
                    print(f"Error evaluating {baseline_name} {dataset_name} {file}: {e}")
                    continue
                
                # Accumulate results
                for mode, count in local_minima_accuracy.items():
                    total_accuracy[mode] += 100 if count else 0
            
            all_results[baseline_name] = total_accuracy
            
            # Accumulate to overall
            for mode, count in total_accuracy.items():
                overall_accuracy[baseline_name][mode] += count
            total_items_overall[baseline_name][dataset_name] += len(output_files)
        
        # Print per-dataset results
        print_results_table(
            args.baseline_name,
            all_results,
            {b: total_items_overall[b][dataset_name] for b in args.baseline_name},
            columns
        )
        print()
    
    # Print overall summary
    print("=" * 31)
    print("OVERALL (across all datasets)")
    print("=" * 31)
    
    baseline_col_width = max(len(b) for b in args.baseline_name) + 2
    col_width = max(len(c) for c in columns) + 2
    
    header = "Baseline".ljust(baseline_col_width) + "".join(c.rjust(col_width) for c in columns)
    print(header)
    print("-" * len(header))
    
    for baseline_name in args.baseline_name:
        row = baseline_name.ljust(baseline_col_width)
        total_num = sum(total_items_overall[baseline_name].values())
        res = overall_accuracy[baseline_name]
        
        if total_num > 0:
            acc = res["Correct"] / total_num
            uar = res["Underverified"] / total_num
            row += str(round(acc, 1)).rjust(col_width) + ","
            row += str(round(uar, 1)).rjust(col_width) + ","
            for c in columns[2:]:
                row += str(round(res[c] / total_num, 1)).rjust(col_width) + ","
        else:
            row += ",".join("0".rjust(col_width) for _ in columns)
        print(row)
    print()
    
    print(total_items_overall)


if __name__ == "__main__":
    args = get_args()
    main(args)
