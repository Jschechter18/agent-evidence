"""Serializable role metadata; missing runtime facts remain null."""
import subprocess
import torch


def role_metadata(agent):
    spec = getattr(agent, "model_spec", None)
    if not isinstance(spec, dict):
        spec = {}
    model = getattr(agent, "model", None)
    config = getattr(model, "config", None)
    def scalar(value):
        return value if isinstance(value, (str, int, float, bool, list, dict)) else None
    dtype = getattr(model, "dtype", None)
    device = getattr(model, "device", None)
    return {
        "model_id": spec.get("id") or scalar(getattr(config, "_name_or_path", None)),
        "requested_revision": spec.get("revision"),
        "resolved_revision": scalar(getattr(config, "_commit_hash", None)),
        "architecture": scalar(getattr(config, "architectures", None)),
        "loader": spec.get("loader"),
        "dtype": str(dtype) if isinstance(dtype, torch.dtype) else None,
        "device": str(device) if isinstance(device, (str, torch.device)) else None,
        "device_map": scalar(getattr(model, "hf_device_map", None)),
        "requested_placement": spec.get("device_map", spec.get("device")),
        "quantization": scalar(getattr(config, "quantization_config", None)),
        "generation": spec.get("generation", {"max_new_tokens": agent.max_new_tokens, "do_sample": False}),
    }


def environment_metadata():
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {
        "git_dirty": bool(result.stdout.strip()) if result.returncode == 0 else None,
        "cuda_version": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else [],
    }
