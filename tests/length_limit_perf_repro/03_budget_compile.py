#!/usr/bin/env python3
import argparse
import inspect
import statistics
import time

import xgrammar as xgr

parser = argparse.ArgumentParser()
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--vocab-size", type=int, default=50_000)
parser.add_argument("--token-cap", type=int, default=32)
parser.add_argument("--runs", type=int, default=7)
args = parser.parse_args()

vocab = ["<think>", "</think>", "<begin>", "<end>", "a", "bb", "中", "<eos>"]
for i in range(args.vocab_size - len(vocab)):
    stem = f"q{i:06x}"
    length = 8 + i % (max(8, args.token_cap) - 7)
    vocab.append(stem + "z" * (length - len(stem)))
tokenizer = xgr.TokenizerInfo(vocab, stop_token_ids=[7])

compiler_kwargs = {"max_threads": 1, "cache_enabled": False}
if args.dynamic:
    if "enable_dynamic_compilation" not in inspect.signature(xgr.GrammarCompiler).parameters:
        raise SystemExit("This XGrammar build does not support dynamic compilation")
    compiler_kwargs["enable_dynamic_compilation"] = True

cases = [
    ("unbounded", {}),
    ("max_tokens=1", {"max_tokens": 1}),
    ("max_tokens=1000000", {"max_tokens": 1_000_000}),
    ("max_chars=1", {"max_chars": 1}),
    ("max_chars=1000000", {"max_chars": 1_000_000}),
]

print("case,median_compile_ms,memory_bytes")
for name, budget in cases:
    content = {"type": "any_text", **budget}
    spec = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "<think>",
            "content": content,
            "end": "</think>",
        },
    }
    xgr.GrammarCompiler(tokenizer, **compiler_kwargs).compile_structural_tag(spec)
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        compiled = xgr.GrammarCompiler(tokenizer, **compiler_kwargs).compile_structural_tag(spec)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    print(f"{name},{statistics.median(samples):.3f},{compiled.memory_size_bytes}")
