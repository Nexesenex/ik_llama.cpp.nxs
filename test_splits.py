import subprocess
import time
import sys
import os
import shutil
from datetime import datetime
import re

ts_values = ["24,24,12", "24,24,13", "24,24,14", "24,24,15", "24,24,16"]
sasf_values = ["0.5", "0.33", "0.25", "0.20", "6", "7"]
smf_values = ["1.0", "2.0", "0.5"]  # split_memory_factor values

exe = r"Q:\GitHub\ik_llama.cpp.fks\out\build\x64_Rel_MSVC_Cuda_Test\bin\llama-server"
model = r"X:\text-generation-webui\models\Qwen3.5-397B-A17B\Qwen3.5-397B-A17B-IQ4_XS-00001-of-01099.gguf"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = r"Q:\GitHub\ik_llama.cpp.fks\test_results_%s.txt" % timestamp

EXEC_TIME = 3
INTERVAL_TIME = 3

def find_last_completed():
    """Find the most recent result file and return the last completed iteration."""
    import glob
    pattern = r"Q:\GitHub\ik_llama.cpp.fks\test_results_*.txt"
    files = glob.glob(pattern)
    if not files:
        return 0
    latest = max(files)
    print("Found existing log: %s" % latest)
    
    with open(latest, 'r', encoding='utf-8') as f:
        content = f.read()
    
    matches = re.findall(r'OUTPUT END \((\d+)/(\d+)\)', content)
    if matches:
        last = max(int(m[0]) for m in matches)
        total = int(matches[0][1])
        print("Resuming from iteration %d/%d" % (last + 1, total))
        return last, latest, total
    return 0, None, None

result = find_last_completed()
start_iter = result[0]
resume_file = result[1]
expected_total = result[2] if result[2] else 0

if start_iter > 0 and expected_total > 0 and start_iter >= expected_total:
    print("Previous run complete, starting fresh")
    start_iter = 0
    resume_file = None
    expected_total = 0
elif start_iter > 0 and resume_file:
    print("Backing up previous log...")
    shutil.copy(resume_file, resume_file + ".bak")
    log_file = resume_file

base_args = [
    "-m", model,
    "-t", "18",
    "-ngl", "150",
    "-sm", "tenpar",
    "-smtps",
    "-ub", "256",
    "-b", "512",
    "-mg", "0",
    "--device", "CUDA0,CUDA1,CUDA2",
    "--max-gpu-per-split", "2",
    "-fa", "1",
    "-cuda", "fusion=1,offload-batch-size=128,mmq-id-size=128,enable-p2p=0,fa-offset=0.6931f",
    "-ot", "^output.weight$=CUDA2",
    "-ncmoe", "45",
    "-no-ooae",
    "-mqkv",
    "-muge",
    "-gr",
    "-ger",
    "-grt", "f16",
    "-rcache",
    "--override-kv", "qwen35moe.expert_used_count=int:10",
    "--chat-template-file", r"Q:\LLAMA_IK_CUSTOM\models\templates\Qwen3.5.jinja",
    "--jinja",
    "-ser", "9,0.2",
    "-c", "163840",
    "-ctk", "q8_0",
    "-ctv", "q5_0",
    "-khad",
    "--context-shift", "1",
    "--host", "127.0.0.1",
    "--port", "8080",
    "-cram", "8",
    "-cram-n-min", "999999",
    "--reasoning-tokens", "none",
    "--minilog"
]

results = []
total = len(ts_values) * len(sasf_values) * len(smf_values) * 2 * 2  # 2 for mota, 2 for sava
count = 0

def log(msg):
    print(msg)
    results.append(msg)

log("Starting tests: %d combinations" % total)
log("=" * 60)

if start_iter > 0:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n=== RESUMING FROM ITERATION %d ===\n" % (start_iter + 1))

with open(log_file, "a" if start_iter > 0 else "w", encoding="utf-8") as f:
    if start_iter == 0:
        f.write("Starting tests: %d combinations\n" % total)
        f.write("Parameters: ts x sasf x smf x mota x sava\n")
        f.write("=" * 60 + "\n")

for ts in ts_values:
    for sasf in sasf_values:
        for smf in smf_values:
            for mota in [False, True]:
                for sava in [False, True]:
                    count += 1
                    if count <= start_iter:
                        continue
                    mota_str = "mota" if mota else "no-mota"
                    sava_str = "sava" if sava else "no-sava"
                    log("Test %d/%d: ts=%s sasf=%s smf=%s %s %s" % (count, total, ts, sasf, smf, mota_str, sava_str))
                    
                    args = base_args + ["-ts", ts, "-sasf", sasf, "-smf", smf]
                    if mota:
                        args.append("-mota")
                    if sava:
                        args.append("-sava")
            
            cmd = [exe] + args
            cmd_str = " ".join(cmd)
            log("Command: %s" % cmd_str)
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n=== COMMAND USED (%d/%d) ===\n" % (count, total))
                f.write(cmd_str + "\n")
                f.write("=== OUTPUT START (%d/%d) ===\n" % (count, total))
            
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                time.sleep(EXEC_TIME)
                proc.terminate()
                try:
                    output = proc.communicate(timeout=1)[0].decode('utf-8', errors='replace')
                except:
                    output = proc.communicate()[0].decode('utf-8', errors='replace') if proc.communicate()[0] else ""
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(output if output else "(no output)")
                    f.write("\n=== OUTPUT END (%d/%d) ===\n" % (count, total))
                
                log("Completed: %s %s" % (mota_str, sava_str))
            except Exception as e:
                log("Error: %s" % str(e))
            
            time.sleep(INTERVAL_TIME)

log("=" * 60)
log("All tests completed!")

with open(log_file, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 60 + "\n")
    f.write("All tests completed!\n")

print("\nResults saved to: " + log_file)
