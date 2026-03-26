import re
import csv
import os
import sys

if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "Q:\\GitHub\\ik_llama.cpp.fks\\test_results_20260326_182628.txt"

timestamp_match = re.search(r"_(\d{8}_\d{6})\.txt$", input_file)
timestamp = timestamp_match.group(1) if timestamp_match else ""

output_file = input_file.replace(".txt", "_parsed.csv")
if timestamp:
    base = input_file.replace(".txt", "").replace(f"_{timestamp}", "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    output_file = os.path.join(os.path.dirname(input_file), f"{base}_parsed_{timestamp}.csv")

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

results = []

command_blocks = content.split("=== COMMAND USED")
total_pattern = re.compile(r"Adjusted splits \(total\)\s+:\s+GPU0:\s+([\d.]+)\s+;\s+GPU1:\s+([\d.]+)\s+;\s+GPU2:\s+([\d.]+)")
buffer_pattern = re.compile(r"Device\s+0:\s+([\d.]+)\s+MiB.*?Device\s+1:\s+([\d.]+)\s+MiB.*?Device\s+2:\s+([\d.]+)\s+MiB", re.DOTALL)

for i, block in enumerate(command_blocks[1:], 1):
    ts_match = re.search(r"-ts\s+([\d,]+)", block)
    sasf_match = re.search(r"-sasf\s+([\d.]+)", block)
    mota_match = re.search(r"-mota", block)
    
    if not (ts_match and sasf_match):
        continue
    
    ts_values = ts_match.group(1).split(",")
    sasf_value = float(sasf_match.group(1))
    mota = "yes" if mota_match else "no"
    
    output_start_idx = block.find("=== OUTPUT START")
    if output_start_idx == -1:
        continue
    
    output_section = block[output_start_idx:]
    total_match = total_pattern.search(output_section)
    buffer_match = buffer_pattern.search(output_section)
    
    if not total_match:
        continue
    
    gpu0 = float(total_match.group(1))
    gpu1 = float(total_match.group(2))
    gpu2 = float(total_match.group(3))
    
    if buffer_match:
        buf_gpu0 = float(buffer_match.group(1))
        buf_gpu1 = float(buffer_match.group(2))
        buf_gpu2 = float(buffer_match.group(3))
    else:
        buf_gpu0 = buf_gpu1 = buf_gpu2 = 0.0
    
    results.append({
        "timestamp": timestamp,
        "test_id": i,
        "total_tests": 0,
        "ts_gpu0": int(ts_values[0]),
        "ts_gpu1": int(ts_values[1]),
        "ts_gpu2": int(ts_values[2]),
        "sasf": sasf_value,
        "mota": mota,
        "split_gpu0": gpu0,
        "split_gpu1": gpu1,
        "split_gpu2": gpu2,
        "total_splits": gpu0 + gpu1 + gpu2,
        "buf_gpu0_mib": buf_gpu0,
        "buf_gpu1_mib": buf_gpu1,
        "buf_gpu2_mib": buf_gpu2
    })

fieldnames = ["timestamp", "test_id", "total_tests", "ts_gpu0", "ts_gpu1", "ts_gpu2", "sasf", "mota", "split_gpu0", "split_gpu1", "split_gpu2", "total_splits", "buf_gpu0_mib", "buf_gpu1_mib", "buf_gpu2_mib"]

total_count = len(results)
for r in results:
    r["total_tests"] = total_count

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Parsed {len(results)} tests to {output_file}")
