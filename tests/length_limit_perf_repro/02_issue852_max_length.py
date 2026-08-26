#!/usr/bin/env python3
import argparse
import inspect
import json
import statistics
import time

import xgrammar as xgr
from transformers import AutoTokenizer

FRAGMENT = (
    "const stockOutQuantity = Number(event.target.value); "
    "if (stockOutQuantity > 0) { setQuantity(stockOutQuantity); } "
    "const response = await fetch(apiBaseUrl + inventoryPath, "
    "{ method: post, body: formData }); const parsed = await response.json(); "
    "navigateTo(inventoryListPage, parsed.identifier);"
)
TEXT = '{"content": "' + " ".join([FRAGMENT] * 4) + '"}'


def schema(max_length):
    content = {"type": "string"}
    if max_length is not None:
        content["maxLength"] = max_length
    return {"type": "object", "properties": {"content": content}, "required": ["content"]}


def replay(name, schema_obj):
    kwargs = {}
    if args.dynamic:
        if "enable_dynamic_compilation" not in inspect.signature(xgr.GrammarCompiler).parameters:
            raise SystemExit("This XGrammar build does not support dynamic compilation")
        kwargs["enable_dynamic_compilation"] = True
    start = time.perf_counter_ns()
    compiled = xgr.GrammarCompiler(tokenizer_info, **kwargs).compile_json_schema(
        json.dumps(schema_obj)
    )
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    fills = []
    for position, token_id in enumerate(token_ids):
        start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(bitmask)
        fills.append((time.perf_counter_ns() - start) / 1e6)
        assert matcher.accept_token(token_id), f"rejected at token {position}"
    print(f"{name},{compile_ms:.3f},{sum(fills):.3f},{statistics.median(fills):.6f}")


parser = argparse.ArgumentParser()
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--model", default="moonshotai/Kimi-K3")
parser.add_argument("--revision", default="a590ce090cb049c93a33dfe8c208ec652aa20503")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model, revision=args.revision, trust_remote_code=True
)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
token_ids = tokenizer.encode(TEXT, add_special_tokens=False)
assert tokenizer.decode(token_ids) == TEXT and len(token_ids) == 273

print("case,compile_ms,fill_total_ms,fill_p50_ms")
replay("maxLength=100000", schema(100_000))
replay("without_maxLength", schema(None))
