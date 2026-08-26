#!/usr/bin/env python3
import argparse
import inspect
import statistics
import time

import xgrammar as xgr

parser = argparse.ArgumentParser()
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[1_000, 5_000, 20_000, 50_000])
parser.add_argument("--token-cap", type=int, default=32)
parser.add_argument("--max-chars", type=int, default=128)
parser.add_argument("--runs", type=int, default=30)
args = parser.parse_args()

compiler_kwargs = {"max_threads": 1, "cache_enabled": False}
if args.dynamic:
    if "enable_dynamic_compilation" not in inspect.signature(xgr.GrammarCompiler).parameters:
        raise SystemExit("This XGrammar build does not support dynamic compilation")
    compiler_kwargs["enable_dynamic_compilation"] = True

print("vocab_size,remaining_1_fill_us")
for vocab_size in args.vocab_sizes:
    vocab = ["<think>", "</think>", "<begin>", "<end>", "a", "bb", "中", "<eos>"]
    for i in range(vocab_size - len(vocab)):
        stem = f"q{i:06x}"
        length = 8 + i % (max(8, args.token_cap) - 7)
        vocab.append(stem + "z" * (length - len(stem)))
    tokenizer = xgr.TokenizerInfo(vocab, stop_token_ids=[7])
    spec = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "<think>",
            "content": {"type": "any_text", "max_chars": args.max_chars},
            "end": "</think>",
        },
    }
    compiled = xgr.GrammarCompiler(tokenizer, **compiler_kwargs).compile_structural_tag(spec)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    for _ in range(args.max_chars - 1):
        assert matcher.accept_token(4)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
    for _ in range(5):
        matcher.fill_next_token_bitmask(bitmask)
    fills = []
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(bitmask)
        fills.append((time.perf_counter_ns() - start) / 1_000)
    print(f"{vocab_size},{statistics.median(fills):.3f}")
