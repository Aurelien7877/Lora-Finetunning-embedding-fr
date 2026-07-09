"""Patch minimal Unsloth GRPO + LFM2 sous Windows (sans vLLM).

À exécuter avant GRPOTrainer si le cache Unsloth est régénéré.
Les variables d'environnement doivent être définies AVANT from_pretrained (cellule GPU).
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

LFM2_SAFE_HIDDEN_FN = '''@torch.compiler.disable(recursive = False)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    # LFM2-safe: never chunk-matmul hidden@lm_head (breaks when .logits is vocab-sized).
    last_dim = hidden_states.shape[-1]
    hidden_dim = lm_head.shape[1]
    if last_dim == hidden_dim:
        logits = F.linear(hidden_states.to(lm_head.dtype), lm_head)
        if logit_scale_multiply != 0.0:
            logits = logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            logits = logits / logit_scale_divide
        if logit_softcapping != 0.0:
            logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    else:
        logits = hidden_states
    return chunked_selective_log_softmax(logits, index, temperature, chunks)
'''


def _resolve_cache_path(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    if (cache_dir / "UnslothGRPOTrainer.py").exists():
        return cache_dir / "UnslothGRPOTrainer.py"
    alt = Path(__file__).resolve().parent / "unsloth_compiled_cache" / "UnslothGRPOTrainer.py"
    if alt.exists():
        return alt
    return cache_dir / "UnslothGRPOTrainer.py"


def patch_unsloth_grpo_lfm2_logits(cache_dir: str | Path = "unsloth_compiled_cache") -> bool:
    cache_path = _resolve_cache_path(cache_dir)
    if not cache_path.exists():
        print(f"Cache introuvable : {cache_path}")
        return False

    text = cache_path.read_text(encoding="utf-8")
    changed = False

    if "LFM2-safe: never chunk-matmul" not in text:
        text, n = re.subn(
            r"@torch\.(?:compile\([^)]*\)|compiler\.disable\([^)]*\))\s*\ndef chunked_hidden_states_selective_log_softmax\([\s\S]*?\n(?=@torch\.|def calculate_pad_tokens)",
            LFM2_SAFE_HIDDEN_FN + "\n",
            text,
            count=1,
        )
        if n:
            changed = True

    replacements = [
        (
            'os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"\n\n            with _get_inference_mode_context_manager(model):',
            'os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"\n\n            with _get_inference_mode_context_manager(model):',
        ),
        (
            'os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"\n\n    lm_head = trainer.model.get_output_embeddings().weight',
            'os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"\n\n    lm_head = trainer.model.get_output_embeddings().weight',
        ),
        (
            "@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)\ndef chunked_selective_log_softmax",
            "@torch.compiler.disable(recursive = False)\ndef chunked_selective_log_softmax",
        ),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            changed = True

    logits_marker = "# LFM2 etc.: UNSLOTH_RETURN_HIDDEN_STATES returns hidden states"
    if logits_marker not in text and "if pixel_values is None:" in text:
        old_block = """                        if pixel_values is None:
                            logits_chunk = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                pixel_values = pixel_values_chunk,
                                image_grid_thw = image_grid_thw_chunk,
                                pixel_attention_mask = pixel_attention_mask_chunk,
                                image_sizes = image_sizes_chunk,
                                **_extra_vision_kwargs,
                            ).logits

                            completion_input_ids_chunk = input_ids_chunk[
                                :, -(logits_to_keep + max_left_pad) :
                            ]
                            logits_chunk = logits_chunk[
                                :, -(logits_to_keep + max_left_pad + 1) :, :
                            ]
                            logits_chunk = logits_chunk[:, :-1, :]
                            logprobs_chunk = (
                                chunked_hidden_states_selective_log_softmax(
                                    logits_chunk,
                                    lm_head,
                                    completion_input_ids_chunk,
                                    chunks = input_ids_chunk.shape[0] * multiplier,
                                    logit_scale_multiply = logit_scale_multiply,
                                    logit_scale_divide = logit_scale_divide,
                                    logit_softcapping = logit_softcapping,
                                    temperature = temperature,
                                )
                            )"""
        new_block = """                        if pixel_values is None:
                            # LFM2 etc.: UNSLOTH_RETURN_HIDDEN_STATES returns hidden states
                            # but chunked_hidden_states_selective_log_softmax fails under torch.compile.
                            # Force real logits and use the logits-only path.
                            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
                            logits_chunk = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                pixel_values = pixel_values_chunk,
                                image_grid_thw = image_grid_thw_chunk,
                                pixel_attention_mask = pixel_attention_mask_chunk,
                                image_sizes = image_sizes_chunk,
                                **_extra_vision_kwargs,
                            ).logits

                            completion_input_ids_chunk = input_ids_chunk[
                                :, -(logits_to_keep + max_left_pad) :
                            ]
                            logits_chunk = logits_chunk[
                                :, -(logits_to_keep + max_left_pad + 1) :, :
                            ]
                            logits_chunk = logits_chunk[:, :-1, :]
                            logprobs_chunk = chunked_selective_log_softmax(
                                logits_chunk,
                                completion_input_ids_chunk,
                                temperature = temperature,
                                chunks = input_ids_chunk.shape[0] * multiplier,
                            )"""
        if old_block in text:
            text = text.replace(old_block, new_block, 1)
            changed = True

    compute_old = """    def compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk):
        # Hidden states -> lm_head matmul path; raw logits -> skip matmul and
        # skip scale/softcap (model forward already applied them).
        chunks = input_ids_chunk.shape[0] * multiplier
        if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
            return efficient_log_softmax(
                new_hidden_states_chunk,
                lm_head,
                completion_ids,
                chunks = chunks,
                logit_scale_multiply = logit_scale_multiply,
                logit_scale_divide = logit_scale_divide,
                logit_softcapping = logit_softcapping,
                temperature = temperature,
                batch_size = B,
            )
        return chunked_selective_log_softmax(
            new_hidden_states_chunk,
            completion_ids,
            temperature = temperature,
            chunks = chunks,
        )"""
    compute_new = """    def compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk):
        # LFM2: always use logits path — hidden-states matmul breaks under torch.compile.
        chunks = input_ids_chunk.shape[0] * multiplier
        return chunked_selective_log_softmax(
            new_hidden_states_chunk,
            completion_ids,
            temperature = temperature,
            chunks = chunks,
        )"""
    if "LFM2: always use logits path" not in text and compute_old in text:
        text = text.replace(compute_old, compute_new, 1)
        changed = True

    efficient_old = """    def efficient_log_softmax(hidden_states, lm_head, index, chunks=32,
                            logit_scale_multiply=0.0, logit_scale_divide=0.0,
                            logit_softcapping=0.0, temperature=1, batch_size=8):
        if (index.shape[1] <= 1024 and batch_size <= 8) or batch_size==1:"""
    efficient_new = """    def efficient_log_softmax(hidden_states, lm_head, index, chunks=32,
                            logit_scale_multiply=0.0, logit_scale_divide=0.0,
                            logit_softcapping=0.0, temperature=1, batch_size=8):
        # LFM2: if forward already returned logits, never matmul with lm_head.
        if hidden_states.shape[-1] != lm_head.shape[1]:
            return chunked_selective_log_softmax(hidden_states, index, temperature, chunks)
        if (index.shape[1] <= 1024 and batch_size <= 8) or batch_size==1:"""
    if "LFM2: if forward already returned logits" not in text and efficient_old in text:
        text = text.replace(efficient_old, efficient_new, 1)
        changed = True

    vlm_old = """                            # Guard: check if model returned hidden states or logits
                            if logits_chunk.shape[-1] == lm_head.shape[1]:
                                logprobs_chunk = (
                                    chunked_hidden_states_selective_log_softmax(
                                        logits_chunk,
                                        lm_head,
                                        completion_input_ids_chunk,
                                        chunks = input_ids_chunk.shape[0] * multiplier,
                                        logit_scale_multiply = logit_scale_multiply,
                                        logit_scale_divide = logit_scale_divide,
                                        logit_softcapping = logit_softcapping,
                                        temperature = temperature,
                                    )
                                )
                            else:
                                # Model returned logits directly - scaling/softcapping already applied by model forward
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                )"""
    vlm_new = """                            logprobs_chunk = chunked_selective_log_softmax(
                                logits_chunk,
                                completion_input_ids_chunk,
                                temperature = temperature,
                                chunks = input_ids_chunk.shape[0] * multiplier,
                            )"""
    if vlm_old in text:
        text = text.replace(vlm_old, vlm_new, 1)
        changed = True

    text, helper_changed = _inject_missing_grpo_helpers(text)
    changed = changed or helper_changed

    if changed:
        cache_path.write_text(text, encoding="utf-8")
        print(f"Patch Unsloth GRPO LFM2 appliqué → {cache_path}")
    else:
        print("Patch Unsloth GRPO LFM2 déjà à jour")
    return changed


ALIGN_COMPLETION_TOOL_MASK_FN = '''def align_completion_tool_mask(
    tool_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Aligns a raw completion-length tool/env mask with Unsloth's repacked loss mask."""
    if tool_mask is None:
        return completion_mask
    if tool_mask.shape[0] != completion_mask.shape[0]:
        raise ValueError("tool_mask batch size must match completion_mask batch size.")

    tool_mask = tool_mask.to(device=completion_mask.device)
    if tool_mask.shape == completion_mask.shape:
        aligned_tool_mask = tool_mask
    else:
        aligned_tool_mask = align_logprobs_with_mask(
            tool_mask,
            completion_mask,
            pad_value=0,
        )
    return completion_mask * aligned_tool_mask.to(dtype=completion_mask.dtype)
'''


def _inject_missing_grpo_helpers(text: str) -> tuple[str, bool]:
    changed = False
    if "def align_completion_tool_mask(" not in text:
        anchor = "    return padded_logprobs\n\ndef autotune_batch_and_chunks("
        if anchor in text:
            text = text.replace(
                anchor,
                "    return padded_logprobs\n\n" + ALIGN_COMPLETION_TOOL_MASK_FN + "\ndef autotune_batch_and_chunks(",
                1,
            )
            changed = True
    return text, changed


def reload_unsloth_grpo_trainer(cache_dir: str | Path = "unsloth_compiled_cache"):
    """Recharge GRPOTrainer depuis le cache patché (obligatoire après patch en cours de session)."""
    cache_path = _resolve_cache_path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    # Ne pas re-importer unsloth ici : ça régénère le cache et efface le patch.
    if "unsloth" not in sys.modules:
        import unsloth  # noqa: F401

    module_name = "unsloth_compiled_cache.UnslothGRPOTrainer"
    for mod in list(sys.modules):
        if "UnslothGRPOTrainer" in mod:
            del sys.modules[mod]

    spec = importlib.util.spec_from_file_location(module_name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Compléter les helpers Unsloth absents du cache (ex. align_completion_tool_mask)
    try:
        from unsloth_zoo.rl_replacements import RL_REPLACEMENTS
        for name, fn in RL_REPLACEMENTS.items():
            if callable(fn) and not hasattr(mod, name):
                setattr(mod, name, fn)
    except Exception:
        pass

    import trl
    import trl.trainer.grpo_trainer as grpo_module

    trainer_cls = mod.UnslothGRPOTrainer
    config_cls = mod.UnslothGRPOConfig

    trl.GRPOTrainer = trainer_cls
    trl.GRPOConfig = config_cls
    trl.trainer.GRPOTrainer = trainer_cls
    trl.trainer.GRPOConfig = config_cls
    grpo_module.GRPOTrainer = trainer_cls
    grpo_module.GRPOConfig = config_cls

    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception:
        pass

    print(f"GRPOTrainer rechargé depuis {cache_path}")
    return trainer_cls, config_cls


def apply_lfm2_grpo_fix(cache_dir: str | Path = "unsloth_compiled_cache"):
    """Patch le cache puis recharge GRPOTrainer en RAM (après import unsloth/modèle)."""
    if "unsloth" not in sys.modules:
        import unsloth  # noqa: F401
    patch_unsloth_grpo_lfm2_logits(cache_dir)
    return reload_unsloth_grpo_trainer(cache_dir)


if __name__ == "__main__":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    apply_lfm2_grpo_fix()
