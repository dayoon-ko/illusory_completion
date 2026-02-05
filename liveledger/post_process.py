import json 
import os 
import argparse 
from glob import glob
from select import select
# from transformers import AutoTokenizer

def process_data(data):
    messages = data["messages"]
    
    thinking_blocks = []
    query_blocks = []
    results_blocks = []

    if messages[0]["role"] == "system":
        messages = messages[1:]
    if messages[0]["role"] == "user":
        messages = messages[1:]
        
    for message in messages:
        role = message["role"]
        content = message["content"] 
        if role == "assistant":
            thinking = content
            thinking_blocks.append(thinking)
            if "tool_calls" in message:
                tool_call = message["tool_calls"][0]
                func_name = tool_call["function"]["name"]
                if func_name == "search":
                    arguments = json.loads(tool_call["function"]["arguments"])
                    if "query" in arguments:
                        query = json.dumps(arguments["query"])
                    else:
                        query = json.dumps(arguments)
                    query_blocks.append("Search: " + query)
                elif func_name == "browse":
                    arguments = json.dumps(tool_call["function"]["arguments"])
                    query_blocks.append("Browse: " + arguments)
                else:
                    # print(f"Unknown tool name: {func_name}")
                    query_blocks.append("Invalid tool call")
        elif role == "tool":
            tool_response = content 
            results_blocks.append(tool_response)
        elif role == "user":
            continue
    
    if len(thinking_blocks) == len(query_blocks):
        thinking_blocks.append(data["content"])
        
    # Valid check 
    try:
        assert len(query_blocks) == len(results_blocks), f"Length of query ({len(query_blocks)}) and results blocks ({len(results_blocks)}) must be the same"
        assert len(thinking_blocks) == len(query_blocks) + 1, f"Length of thinking ({len(thinking_blocks)}) and query blocks ({len(query_blocks)}) must be the same"
    except Exception as e:
        print(f"Error: {e}")
        # print(json.dumps(messages, indent=4, ensure_ascii=False))
        raise
    
    data["output"] = {"thinking_blocks": thinking_blocks, "query_blocks": query_blocks, "results_blocks": results_blocks}
    
    # Check prediction 
    if "prediction" not in data or data["prediction"] == "":
        data["prediction"] = thinking_blocks[-1]
        
    return data 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", "-d", nargs="+", type=str, default=["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa"]) # browsecomp
    parser.add_argument("--input_dir", type=str, default="outputs_baseline")
    parser.add_argument("--output_dir", type=str, default="../baselines/results/react")
    args = parser.parse_args()
    
    # tokenizer = AutoTokenizer.from_pretrained("hkust-nlp/WebExplorer-8B")

    for dataset in args.datasets:
        
        print(f"Processing {dataset} ...")

        input_dir = args.input_dir
        output_dir = args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        
        input_files = glob(os.path.join(input_dir, f"{dataset}/*.json"))
        input_files = sorted(input_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        output_file = os.path.join(output_dir, f"{dataset}.jsonl")
            
        # Load inputs 
        quest_const_dict = {}
        with open(f"../datasets/{dataset}/test_mcqa.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                q = data["question"]
                c = data["constraints"]
                quest_const_dict[q] = c
                
        outputs_all = []
        total_num = {
            "browsecomp": 99,
            "deepsearchqa": 43,
            "frames": 33,
            "livedrbench": 28,
            "webwalkerqa": 12,
        }
        blank_files = 0
        input_files = [f"{input_dir}/{dataset}/{i}.json" for i in range(total_num[dataset])]
        for input_file in input_files:
            if not os.path.exists(input_file):
                outputs_all.append({"question": "", "answer": "", "content": "", "messages": [], "prediction": ""})
                blank_files += 1
                continue
            with open(input_file, "r") as f:
                data = json.load(f)
            try:
                output = process_data(data)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Input file: {input_file}")
                raise
            data["constraints"] = quest_const_dict[data["question"]]
            outputs_all.append(data)
        print("Total number of outputs: ", len(outputs_all))
        print("Number of blank files: ", blank_files)
        with open(output_file, "w") as f:
            for data in outputs_all:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")