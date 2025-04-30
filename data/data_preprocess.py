import json

# Define input and output file paths directly in the script for easier management
INPUT_FILE = "data/train_merged_shuffled.json"
OUTPUT_FILE = "data/emodata_sharegpt.json"
DATASET_INFO_FILE = "data/dataset_info.json"

def convert_emollm_to_sharegpt():
    """
    Convert EmoLLM data format to ShareGPT format.
    
    The EmoLLM format is:
    {"instruction": "...", "input": "", "output": "..."}
    
    The ShareGPT format is:
    [
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        },
        ...
    ]
    """
    # List to store converted data
    sharegpt_data = []
    
    # Read input file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Process each line (assuming JSONL format)
        for line in f:
            try:
                # Parse JSON line
                item = json.loads(line.strip())
                
                # Create ShareGPT conversation object
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": item["instruction"]
                        },
                        {
                            "from": "gpt",
                            "value": item["output"]
                        }
                    ]
                }
                
                # Add to results list
                sharegpt_data.append(conversation)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
            except KeyError as e:
                print(f"Missing expected key: {e}")
    
    # Write output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. {len(sharegpt_data)} items converted.")
    
    # Create dataset_info.json content
    dataset_info = {
        "emollm_sharegpt": {
            "file_name": OUTPUT_FILE,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        }
    }
    
    # Output dataset info
    with open(DATASET_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"{DATASET_INFO_FILE} created.")

# Execute the conversion when script is run directly
if __name__ == "__main__":
    convert_emollm_to_sharegpt()