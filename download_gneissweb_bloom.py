import os, time
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")), flush=True)

from huggingface_hub import hf_hub_download

REPO_ID = "ibm-granite/GneissWeb.bloom"
FILENAME = "gneissweb.bloom"

print(f"Downloading {FILENAME} (~28GB) from {REPO_ID}...", flush=True)
t0 = time.time()
file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="model")
print(f"Downloaded in {time.time()-t0:.1f}s: {file_path}", flush=True)

size_gb = os.path.getsize(file_path) / (1024**3)
print(f"File size: {size_gb:.2f} GB", flush=True)
