#!/usr/bin/env python3
import argparse
import inspect
import statistics
import time

import xgrammar as xgr

parser = argparse.ArgumentParser()
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--vocab-size", type=int, default=5_000)
parser.add_argument("--token-cap", type=int, default=32)
parser.add_argument("--budgets", type=int, nargs="+", default=[1, 64, 1024, 4096])
parser.add_argument("--runs", type=int, default=100)
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

print("format,budget,boundary_fill_us")
for budget in args.budgets:
    for kind in ("any_text", "any_tokens"):
        if kind == "any_text":
            tag = {
                "type": "tag",
                "begin": "<think>",
                "content": {"type": kind, "max_tokens": budget},
                "end": "</think>",
            }
            begin_token = 0
        else:
            tag = {
                "type": "tag",
                "begin": {"type": "token", "token": 2},
                "content": {"type": kind, "max_tokens": budget},
                "end": {"type": "token", "token": 3},
            }
            begin_token = 2
        spec = {"type": "structural_tag", "format": tag}
        compiled = xgr.GrammarCompiler(tokenizer, **compiler_kwargs).compile_structural_tag(spec)
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_token(begin_token)
        for _ in range(budget):
            assert matcher.accept_token(4)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
        for _ in range(5):
            matcher.fill_next_token_bitmask(bitmask)
        fills = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            matcher.fill_next_token_bitmask(bitmask)
            fills.append((time.perf_counter_ns() - start) / 1_000)
        print(f"{kind},{budget},{statistics.median(fills):.3f}")
