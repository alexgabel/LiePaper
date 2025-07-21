import os
import shutil

BASE_DIR = "saved"
TARGET_LOG_DIR = os.path.join(BASE_DIR, "log")
TARGET_MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure the target log and model dirs exist
os.makedirs(TARGET_LOG_DIR, exist_ok=True)
os.makedirs(TARGET_MODEL_DIR, exist_ok=True)

# Go through each subdir in 'saved'
for subdir in os.listdir(BASE_DIR):
    subpath = os.path.join(BASE_DIR, subdir)

    # Skip if not a directory or is 'log' or 'models'
    if not os.path.isdir(subpath) or subdir in ["log", "models"]:
        continue

    print(f"Processing: {subdir}")

    # Copy log files
    log_src = os.path.join(subpath, "log", subdir)
    if os.path.isdir(log_src):
        for run in os.listdir(log_src):
            run_src = os.path.join(log_src, run)
            run_dst = os.path.join(TARGET_LOG_DIR, subdir, run)
            if not os.path.exists(run_dst):
                shutil.copytree(run_src, run_dst)
                print(f"  -> Copied log: {run_dst}")

    # Copy model files
    model_src = os.path.join(subpath, "models", subdir)
    if os.path.isdir(model_src):
        for run in os.listdir(model_src):
            run_src = os.path.join(model_src, run)
            run_dst = os.path.join(TARGET_MODEL_DIR, subdir, run)
            if not os.path.exists(run_dst):
                shutil.copytree(run_src, run_dst)
                print(f"  -> Copied models: {run_dst}")