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
parser.add_argument("--max-chars", type=int, default=128)
parser.add_argument("--runs", type=int, default=9)
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
points = [x for x in (33, 32, 16, 8, 1) if x <= args.max_chars]
samples = {x: [] for x in points}
window_samples = []

for run in range(args.runs + 1):
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
    fills = {}
    for consumed in range(args.max_chars):
        remaining = args.max_chars - consumed
        start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(bitmask)
        fills[remaining] = (time.perf_counter_ns() - start) / 1_000
        assert matcher.accept_token(4)
    if run:
        for remaining in points:
            samples[remaining].append(fills[remaining])
        window = range(1, min(args.token_cap, args.max_chars) + 1)
        window_samples.append(sum(fills[x] for x in window))

print("metric,value")
for remaining in points:
    print(f"remaining_{remaining}_fill_us,{statistics.median(samples[remaining]):.3f}")
window_size = min(args.token_cap, args.max_chars)
print(f"last_{window_size}_total_ms,{statistics.median(window_samples) / 1_000:.3f}")
