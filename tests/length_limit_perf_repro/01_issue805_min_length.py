#!/usr/bin/env python3
import argparse
import inspect
import json
import statistics
import time

import xgrammar as xgr

BASE62 = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def base62(value: int) -> bytes:
    result = bytearray(b"000000")
    for i in range(5, -1, -1):
        result[i], value = BASE62[value % 62], value // 62
    return bytes(result)


parser = argparse.ArgumentParser()
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--tokens", type=int, default=70_000)
parser.add_argument("--max-prefix", type=int, default=4_096)
parser.add_argument("--content-length", type=int, default=120)
parser.add_argument("--runs", type=int, default=5)
args = parser.parse_args()

vocab = [bytes([i]) for i in range(256)] + [
    b"a" * (1 + i % args.max_prefix) + b'"}' + base62(i) for i in range(args.tokens)
]
vocab.append(b"<eos>")
tokenizer = xgr.TokenizerInfo(
    vocab, xgr.VocabType.RAW, vocab_size=len(vocab), stop_token_ids=[len(vocab) - 1]
)
kwargs = {"cache_enabled": False}
if args.dynamic:
    if "enable_dynamic_compilation" not in inspect.signature(xgr.GrammarCompiler).parameters:
        raise SystemExit("This XGrammar build does not support dynamic compilation")
    kwargs["enable_dynamic_compilation"] = True
compiler = xgr.GrammarCompiler(tokenizer, **kwargs)
prefix = '{"description": "' + "a" * args.content_length

print("minLength,median_fill_ms")
for minimum in (128, 129, 200):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"description": {"type": "string", "minLength": minimum}},
        "required": ["description"],
    }
    compiled = compiler.compile_json_schema(json.dumps(schema), any_whitespace=False)
    samples = []
    for _ in range(args.runs):
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_string(prefix)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
        start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(bitmask)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    print(f"{minimum},{statistics.median(samples):.3f}")
