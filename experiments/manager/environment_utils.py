import platform, subprocess, json, os, sys

def capture_environment(run_dir):
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": get_version("tensorflow"),
        "torch_version": get_version("torch"),
        "cuda_version": get_cuda_version()
    }
    with open(os.path.join(run_dir, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)
    print("[Env] Environment snapshot saved.")

def get_version(pkg):
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"

def get_cuda_version():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        return out.decode().strip()
    except Exception:
        return "not_available"
