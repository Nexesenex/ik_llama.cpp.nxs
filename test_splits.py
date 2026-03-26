import subprocess
import time
import sys
import os
import shutil
from datetime import datetime
import re

ts_values = ["24,24,12", "24,24,13", "24,24,14", "24,24,15", "24,24,16"]
sasf_values = ["0.5", "0.33", "0.25", "0.20", "6", "7"]

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
    
    matches = re.findall(r'OUTPUT END \((\d+)/60\)', content)
    if matches:
        last = max(int(m) for m in matches)
        print("Resuming from iteration %d/60" % (last + 1))
        return last, latest
    return 0, None

start_iter, resume_file = find_last_completed()
if start_iter >= 60:
    print("Previous run complete, starting fresh")
    start_iter = 0
    resume_file = None
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
total = len(ts_values) * len(sasf_values) * 2
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
        f.write("=" * 60 + "\n")

for ts in ts_values:
    for sasf in sasf_values:
        for mota in [False, True]:
            count += 1
            if count <= start_iter:
                continue
            mota_str = "-mota" if mota else "no-mota"
            log("Test %d/%d: ts=%s sasf=%s %s" % (count, total, ts, sasf, mota_str))
            
            args = base_args + ["-ts", ts, "-sasf", sasf]
            if mota:
                args.append("-mota")
            
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
                
                log("Completed: %s" % mota_str)
            except Exception as e:
                log("Error: %s" % str(e))
            
            time.sleep(INTERVAL_TIME)

log("=" * 60)
log("All tests completed!")

with open(log_file, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 60 + "\n")
    f.write("All tests completed!\n")

print("\nResults saved to: " + log_file)
