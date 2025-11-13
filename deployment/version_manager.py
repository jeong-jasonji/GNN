import os, json, datetime

def register_model_version(run_dir, export_path, metrics=None):
    registry_file = os.path.join(run_dir, "model_registry.json")
    version_info = {
        "version_path": export_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics or {}
    }
    registry = []
    if os.path.exists(registry_file):
        registry = json.load(open(registry_file))
    registry.append(version_info)
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[VersionManager] Registered model version at {export_path}")
    return registry_file
