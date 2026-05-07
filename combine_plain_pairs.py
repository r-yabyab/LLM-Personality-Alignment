import os
import glob

input_dir = os.path.join(os.path.dirname(__file__), "data", "transformed", "plain", "plain_pairs")
output_file = os.path.join(os.path.dirname(__file__), "plain_pairs_combined.jsonl")

files = glob.glob(os.path.join(input_dir, "*.jsonl"))

with open(output_file, "w", encoding="utf-8") as out:
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + "\n")

print(f"Combined {len(files)} files -> {output_file}")
