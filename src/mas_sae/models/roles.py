"""Independent model roles with explicit loaders and no precision fallback."""
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mas_sae.agents.base import validate_chat_template_kwargs
from mas_sae.models.loader import load_gemma

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DEFAULT_TOKENS = {"solver": 32, "critic": 128, "validator": 4}
# Spec keys that configure the agent, not the weights. Roles that differ only
# in these keys share one loaded checkpoint.
AGENT_ONLY_KEYS = ("generation", "chat_template_kwargs")


def resolve_role_spec(role, spec, *, default_max_new_tokens):
    """Validate one model-role spec and fill generation defaults."""
    if not isinstance(spec, dict):
        raise ValueError(f"{role} must be a mapping")
    allowed = {"id", "loader", "revision", "dtype", "device_map", "device", "cache_dir",
               "generation", "chat_template_kwargs"}
    if set(spec) - allowed:
        raise ValueError(f"Unsupported {role} options: {set(spec) - allowed}")
    for key in ("id", "loader", "dtype"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"{role}.{key} is required")
    if spec["loader"] not in {"gemma3", "qwen3"} or spec["dtype"] not in DTYPES:
        raise ValueError(f"Unsupported loader or dtype for {role}")
    revision = spec.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError(f"Invalid {role}.revision")
    if "device" in spec and "device_map" in spec:
        raise ValueError("Specify device or device_map, not both")
    placement = spec.get("device_map", spec.get("device"))
    if placement is None or not isinstance(placement, (str, dict)) or not placement:
        raise ValueError(f"Explicit device or device_map required for {role}")
    if not isinstance(spec.get("generation", {}), dict):
        raise ValueError("generation must be a mapping")
    generation = {"do_sample": False, "max_new_tokens": default_max_new_tokens,
                  **spec.get("generation", {})}
    if set(generation) - {"do_sample", "max_new_tokens", "temperature", "top_p"}:
        raise ValueError("Unsupported generation settings")
    if type(generation["do_sample"]) is not bool:
        raise ValueError("do_sample must be boolean")
    if type(generation["max_new_tokens"]) is not int or generation["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    for key in ("temperature", "top_p"):
        if key in generation:
            v = generation[key]
            if not generation["do_sample"] or type(v) not in (int, float) or not math.isfinite(v) or not 0 < v or (key == "top_p" and v > 1):
                raise ValueError(f"Invalid sampling parameter {key}")
    resolved = {**spec, "revision": revision, "generation": generation}
    if "chat_template_kwargs" in spec:
        try:
            resolved["chat_template_kwargs"] = validate_chat_template_kwargs(spec["chat_template_kwargs"])
        except ValueError as error:
            raise ValueError(f"{role}.{error}") from error
    return resolved


def resolve_roles(config):
    if "roles" in config:
        if "model" in config:
            raise ValueError("Use either roles or legacy model, not both")
        raw = config["roles"]
        if not isinstance(raw, dict) or set(raw) != set(DEFAULT_TOKENS):
            raise ValueError("roles must specify solver, critic, validator")
    else:
        model = config.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"].strip():
            raise ValueError("model.id must be a non-empty string")
        raw = {role: {"id": config["model"]["id"], "loader": "gemma3",
                      "revision": None, "dtype": "bfloat16", "device_map": "auto"}
               for role in DEFAULT_TOKENS}
    return {role: resolve_role_spec(role, spec, default_max_new_tokens=DEFAULT_TOKENS[role])
            for role, spec in raw.items()}


def weight_identity(spec):
    """Canonical key of the weights a spec loads, ignoring agent-only settings."""
    return json.dumps({k: v for k, v in spec.items() if k not in AGENT_ONLY_KEYS}, sort_keys=True)


def load_spec(spec):
    common = {"cache_dir": spec.get("cache_dir", "checkpoints/huggingface"),
              "revision": spec["revision"]}
    placement = {"device_map": spec.get("device_map", {"": spec.get("device")})}
    if spec["loader"] == "gemma3":
        return load_gemma(model_id=spec["id"], dtype=DTYPES[spec["dtype"]], **common, **placement)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], **common)
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], dtype=DTYPES[spec["dtype"]], **common, **placement)
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("qwen3 loader requires a Qwen3 model")
    model.eval()
    return model, tokenizer


def load_role_models(specs):
    cache, roles = {}, {}
    for role, spec in specs.items():
        key = weight_identity(spec)
        if key not in cache:
            cache[key] = load_spec(spec)
        roles[role] = cache[key]
    return roles


def configure_agent(agent, spec):
    agent.max_new_tokens = spec["generation"]["max_new_tokens"]
    agent.generation_settings = {k: v for k, v in spec["generation"].items() if k != "max_new_tokens"}
    agent.text_only = spec["loader"] == "qwen3"
    agent.chat_template_kwargs = dict(spec.get("chat_template_kwargs", {}))
    agent.model_spec = spec
