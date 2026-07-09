"""Isolated one-task MTEB worker for the E5 French notebook.

The parent notebook starts a fresh process for every task.  This file never
updates the final result in place: a result appears only after the complete
task has succeeded and the temporary JSON has been atomically replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Keep native libraries from creating a large CPU thread pool in every worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import mteb
import numpy as np
import psutil
import torch
from mteb.models import EncoderProtocol, ModelMeta
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer


class SafeE5Encoder:
    def __init__(
        self,
        model: SentenceTransformer,
        *,
        model_name: str,
        model_revision: str,
        adapted_from: str,
        batch_size: int,
        chunk_size: int,
        cooldown_seconds: float,
        min_free_ram_gb: float,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.cooldown_seconds = cooldown_seconds
        self.min_free_ram_bytes = int(min_free_ram_gb * 1024**3)

        base_meta = ModelMeta.from_sentence_transformer_model(model)
        self.mteb_model_meta = base_meta.model_copy(
            update={
                "name": model_name,
                "revision": model_revision,
                "adapted_from": adapted_from,
            }
        )

    def _guard_memory(self) -> None:
        available = psutil.virtual_memory().available
        if available < self.min_free_ram_bytes:
            raise MemoryError(
                f"RAM libre sous la limite de sécurité: {available / 1024**3:.2f} Go"
            )

    def _encode_texts(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        total = len(texts)
        dim = self.model.get_embedding_dimension()
        output = np.empty((total, dim), dtype=np.float32)

        for start in range(0, total, self.chunk_size):
            self._guard_memory()
            chunk = texts[start : start + self.chunk_size]
            chunk = [text if text.startswith(prefix) else prefix + text for text in chunk]
            embeddings = self.model.encode(
                chunk,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            output[start : start + len(chunk)] = np.asarray(embeddings, dtype=np.float32)
            del embeddings
            if self.cooldown_seconds:
                time.sleep(self.cooldown_seconds)
        return output

    def encode(self, inputs, *, prompt_type=None, **kwargs) -> np.ndarray:
        prefix = "passage: " if prompt_type == PromptType.document else "query: "
        batch_size = kwargs.get("batch_size", self.batch_size)

        if isinstance(inputs, (list, tuple)):
            return self._encode_texts(list(inputs), prefix, batch_size)

        total = len(inputs.dataset)
        dim = self.model.get_embedding_dimension()
        output = np.empty((total, dim), dtype=np.float32)
        buffer: list[str] = []
        cursor = 0

        for batch in inputs:
            buffer.extend(batch["text"])
            if len(buffer) >= self.chunk_size:
                embeddings = self._encode_texts(buffer, prefix, batch_size)
                output[cursor : cursor + len(buffer)] = embeddings
                cursor += len(buffer)
                buffer.clear()
        if buffer:
            embeddings = self._encode_texts(buffer, prefix, batch_size)
            output[cursor : cursor + len(buffer)] = embeddings
            cursor += len(buffer)

        if cursor != total:
            raise RuntimeError(f"Encodages incomplets: {cursor}/{total}")
        return output

    def encode_queries(self, queries, **kwargs) -> np.ndarray:
        return self._encode_texts(
            list(queries), "query: ", kwargs.get("batch_size", self.batch_size)
        )

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        if isinstance(corpus, dict):
            raw_texts = corpus["text"]
            titles = corpus.get("title", [""] * len(raw_texts))
            texts = [
                (title + " " if title else "") + text
                for title, text in zip(titles, raw_texts)
            ]
        else:
            texts = [
                (doc.get("title", "") + " " if doc.get("title") else "")
                + doc["text"]
                for doc in corpus
            ]
        return self._encode_texts(
            texts, "passage: ", kwargs.get("batch_size", self.batch_size)
        )

    def similarity(self, embeddings1, embeddings2):
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1, embeddings2):
        return self.model.similarity_pairwise(embeddings1, embeddings2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapted-from", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--benchmark-label", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--cooldown-seconds", type=float, default=0.25)
    parser.add_argument("--min-free-ram-gb", type=float, default=0.75)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(2, args.cpu_threads))
    if os.name == "nt":
        try:
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except (psutil.AccessDenied, AttributeError):
            pass
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(result_path.suffix + f".{os.getpid()}.tmp")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible")

    model = SentenceTransformer(
        args.model_dir,
        device=args.device,
        model_kwargs={"torch_dtype": torch.float32},
    )
    encoder = SafeE5Encoder(
        model,
        model_name=args.model_label,
        model_revision=args.model_revision,
        adapted_from=args.adapted_from,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        cooldown_seconds=args.cooldown_seconds,
        min_free_ram_gb=args.min_free_ram_gb,
    )
    if not isinstance(encoder, EncoderProtocol):
        raise TypeError("SafeE5Encoder ne satisfait pas EncoderProtocol")

    tasks = mteb.get_tasks(tasks=[args.task], languages=["fra"])
    results = mteb.evaluate(
        encoder,
        tasks=tasks,
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,
        overwrite_strategy="always",
    )
    score = float(results[0].get_score())
    payload = {
        "run_name": args.run_name,
        "task": args.task,
        "score": score,
        "model_label": args.model_label,
        "model_revision": args.model_revision,
        "model_path": str(Path(args.model_dir).resolve()),
        "benchmark_label": args.benchmark_label,
        "eval_device": args.device,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "normalize_embeddings": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, result_path)
    print(json.dumps({"task": args.task, "score": score, "status": "ok"}))


if __name__ == "__main__":
    main()
