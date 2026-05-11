#!/usr/bin/env python3
"exec" "python3" "$0" "$@"
"""
Start vLLM backend in a detached subprocess with proper logging.
"""
import subprocess
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "vllm_backend.log")

os.makedirs(LOG_DIR, exist_ok=True)

print(f"Starting vLLM backend, logging to {LOG_FILE}")

with open(LOG_FILE, "w") as f:
    f.write(f"=== Starting vLLM at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.flush()

venv_python = os.path.join(SCRIPT_DIR, "venv_vllm", "bin", "python3")
if not os.path.exists(venv_python):
    venv_python = sys.executable

proc = subprocess.Popen(
    [venv_python, "zaya_server.py"],
    cwd=SCRIPT_DIR,
    stdin=open(os.devnull, "r"),
    stdout=open(LOG_FILE, "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)

print(f"Started with PID: {proc.pid}")

for i in range(10):
    time.sleep(5)
    if proc.poll() is not None:
        print(f"Process died with exit code: {proc.returncode}")
        break
    with open(LOG_FILE) as f:
        lines = f.readlines()
    print(f"[{i+1}/10] Still running. Last 5 lines:")
    for line in lines[-5:]:
        print(f"  {line.rstrip()}")
