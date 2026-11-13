# Copyright (c) 2026 by Contributors
import json

import pytest

import xgrammar as xgr


def compile_object(schema, texts, style="json", dynamic=False, any_order=True):
    vocab = [bytes([i]) for i in range(256)] + [s.encode() for s in texts] + [b"<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiler = xgr.GrammarCompiler(info, max_threads=8, enable_dynamic_compilation=dynamic)
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "json_schema",
            "style": style,
            "json_schema": schema,
            "any_order": any_order,
        },
    }
    return compiler.compile_structural_tag(json.dumps(tag)), info


def render(pairs, style):
    if style in ("json", "gemma"):
        names = [(json.dumps(k, ensure_ascii=False) if style == "json" else k, v) for k, v in pairs]
        return "{" + ",".join(k + ":" + str(v) for k, v in names) + "}"
    if style == "qwen_xml":
        return "".join(f"<parameter={k}>{v}</parameter>" for k, v in pairs)
    if style == "minimax_xml":
        return "".join(f'<parameter name="{k}">{v}</parameter>' for k, v in pairs)
    if style in ("deepseek_xml", "deepseek_v4_1_xml"):
        space = " " if style == "deepseek_v4_1_xml" else ""
        return "".join(
            f'<｜DSML｜{space}parameter name="{k}" string="false">{v}</｜DSML｜{space}parameter>'
            for k, v in pairs
        )
    if style == "glm_xml":
        return "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in pairs)
    if style == "cohere_xml":
        return "".join(f'<cofl:value name="{k}" type="json">{v}</cofl:value>' for k, v in pairs)
    if style == "kimi_k3_xml":
        return "".join(
            f'<|open|>argument key="{k}" type="number"<|sep|>{v}<|close|>argument<|sep|>'
            for k, v in pairs
        )
    assert style == "minimax_m3_xml"
    return "".join(f"]<]minimax[>[<{k}>{v}]<]minimax[>[</{k}>" for k, v in pairs)


def check_path(compiled, info, ids, expected):
    matcher = xgr.GrammarMatcher(compiled)
    reference = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    rejected = False
    for token in ids:
        matcher.fill_next_token_bitmask(mask)
        allowed = bool(int(mask[0, token // 32]) & (1 << (token % 32)))
        direct = reference.accept_token(token)
        assert allowed == direct
        assert matcher.accept_token(token) == direct
        if not direct:
            rejected = True
            break
    assert (not rejected and matcher.is_terminated()) == expected


def check_next_token(matcher, info, token, expected):
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    matcher.fill_next_token_bitmask(mask)
    allowed = bool(int(mask[0, token // 32]) & (1 << (token % 32)))
    assert allowed == matcher.fork().accept_token(token) == expected
    assert matcher.accept_token(token) == expected


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("pattern", [None, "^(a|az)$"])
def test_qwen_dynamic_name_followup_and_duplicate_prefix(dynamic, pattern):
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    if pattern is not None:
        schema["propertyNames"] = {"pattern": pattern}
    texts = ["aé", "a!", "a>1</parameter>", ">1</para", "meter>", "z>2</parameter>"]
    compiled, info = compile_object(schema, texts, "qwen_xml", dynamic)
    stop = info.vocab_size - 1
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        prefix = list(b"<parameter=")
        for invalid in (256, 257):
            matcher = xgr.GrammarMatcher(grammar)
            assert all(matcher.accept_token(token) for token in prefix)
            check_next_token(matcher, info, invalid, False)
        # A single token can finish the name, delimiter, value and closing tag.
        check_path(grammar, info, prefix + [258, stop], True)
        # The closing tag can instead span tokens, independently of key completion.
        check_path(grammar, info, prefix + [ord("a"), 259, 260, stop], True)

        matcher = xgr.GrammarMatcher(grammar)
        first = b"<parameter=a>1</parameter><parameter=a"
        assert all(matcher.accept_token(token) for token in first)
        # The repeated short name still has a distinct longer continuation.
        check_next_token(matcher, info, ord(">"), False)
        check_next_token(matcher, info, 261, True)
        check_next_token(matcher, info, stop, True)
        assert matcher.is_terminated()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("first", ["a", "b"])
def test_qwen_followup_lookahead_preserves_guarded_lazy_extension(dynamic, first):
    texts = ["xa>1", "xaz>1"]
    vocab = [bytes([i]) for i in range(256)] + [text.encode() for text in texts] + [b"<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[258])
    source = (
        'root[object_keys] ::= first key ">1"\n'
        f'first[object_key="{first}"] ::= "x"\n'
        'key[lazy, object_key_raw] ::= ("a" | "az") (=">")\n'
    )
    compiled = xgr.GrammarCompiler(
        info, max_threads=1, enable_dynamic_compilation=dynamic
    ).compile_grammar(source)
    for text, token, expected in ((texts[0], 256, first != "a"), (texts[1], 257, first == "a")):
        check_path(compiled, info, list(text.encode()) + [258], expected)
        check_path(compiled, info, [token, 258], expected)


@pytest.mark.parametrize("style", ["json", "qwen_xml", "minimax_xml", "deepseek_xml", "cohere_xml"])
@pytest.mark.parametrize("any_order", [False, True])
def test_dynamic_name_followup_renderer_and_order_controls(style, any_order):
    schema = {
        "type": "object",
        "patternProperties": {"^[az]+$": {"type": "integer"}},
        "additionalProperties": False,
    }
    unique = render([("a", 1), ("az", 2)], style)
    duplicate = render([("a", 1), ("a", 2)], style)
    compiled, info = compile_object(schema, [unique, duplicate], style, any_order=any_order)
    for text, token, expected in ((unique, 256, True), (duplicate, 257, not any_order)):
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [token, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "key, encoded",
    [
        ("a&b", "a&amp;b"),
        ("a&amp;b", "a&amp;amp;b"),
        ("a<b", "a&lt;b"),
        ("a>b", "a&gt;b"),
        ('a"b', "a&quot;b"),
        ("é", "é"),
    ],
)
def test_cohere_declared_and_pattern_names_share_raw_identity(key, encoded, dynamic):
    schema = {
        "type": "object",
        "properties": {key: {"type": "integer"}},
        "patternProperties": {"^" + encoded + "$": {"type": "integer"}},
        "required": [key],
        "additionalProperties": False,
    }
    one = f'<cofl:value name="{encoded}" type="json">1</cofl:value>'
    two = f'<cofl:value name="{encoded}" type="json">2</cofl:value>'
    texts = [one, one + one, one + two]
    compiled, info = compile_object(schema, texts, "cohere_xml", dynamic)
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        for index, text in enumerate(texts):
            expected = index == 0
            check_path(grammar, info, list(text.encode()) + [info.vocab_size - 1], expected)
            check_path(grammar, info, [256 + index, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_cohere_key_identity_does_not_decode_entities(dynamic):
    keys = ["a&b", "a&amp;b"]
    schema = {
        "type": "object",
        "properties": {key: {"type": "integer"} for key in keys},
        "required": keys,
        "additionalProperties": False,
    }
    unique = render([("a&amp;b", 1), ("a&amp;amp;b", 2)], "cohere_xml")
    duplicate = unique + render([("a&amp;b", 3)], "cohere_xml")
    compiled, info = compile_object(schema, [unique, duplicate], "cohere_xml", dynamic)
    for text, token, expected in ((unique, 256, True), (duplicate, 257, False)):
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [token, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("any_order", [False, True])
@pytest.mark.parametrize(
    "key, encoded",
    [("a&b", "a&amp;b"), ('a"b', "a&quot;b"), ("a&amp;b", "a&amp;amp;b")],
)
def test_kimi_declared_and_pattern_names_share_encoded_identity(key, encoded, any_order, dynamic):
    schema = {
        "type": "object",
        "properties": {key: {"type": "integer"}},
        "patternProperties": {"^" + encoded + "$": {"type": "integer"}},
        "additionalProperties": False,
    }
    one = render([(encoded, 1)], "kimi_k3_xml")
    two = render([(encoded, 2)], "kimi_k3_xml")
    texts = [one, one + one, one + two]
    compiled, info = compile_object(schema, texts, "kimi_k3_xml", dynamic, any_order=any_order)
    # Fixed-order pattern alternatives retain their existing repetition behavior.
    for index, text in enumerate(texts):
        expected = index == 0 or not any_order
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [256 + index, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_kimi_encoded_identity_keeps_literal_entities_distinct(dynamic):
    schema = {
        "type": "object",
        "properties": {key: {"type": "integer"} for key in ("a&b", "a&amp;b")},
        "patternProperties": {"^(a&amp;b|a&amp;amp;b)$": {"type": "integer"}},
        "additionalProperties": False,
    }
    first = render([("a&amp;b", 1)], "kimi_k3_xml")
    second = render([("a&amp;amp;b", 2)], "kimi_k3_xml")
    texts = [first + second, second + first, first + second + first]
    compiled, info = compile_object(schema, texts, "kimi_k3_xml", dynamic)
    for index, text in enumerate(texts):
        expected = index < 2
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [256 + index, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_kimi_nested_json_key_identity_does_not_escape_attributes(dynamic):
    inner = {
        "type": "object",
        "properties": {key: {"type": "integer"} for key in ("a&b", "a&amp;b")},
        "patternProperties": {"^a&b$": {"type": "integer"}},
        "additionalProperties": False,
    }
    schema = {"type": "object", "properties": {"payload": inner}, "additionalProperties": False}
    prefix = '<|open|>argument key="payload" type="object"<|sep|>'
    suffix = "<|close|>argument<|sep|>"
    bodies = ['{"a&b":1,"a&amp;b":2}', '{"a&b":1,"a&b":2}']
    texts = [prefix + body + suffix for body in bodies]
    compiled, info = compile_object(schema, texts, "kimi_k3_xml", dynamic)
    for index, text in enumerate(texts):
        expected = index == 0
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [256 + index, info.vocab_size - 1], expected)


@pytest.mark.parametrize("style", ["json", "qwen_xml", "minimax_xml"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("size", [0, 1, 2, 8, 64, 65, 128])
def test_declared_any_order_key_identity(size, style, dynamic):
    keys = [f"k{i}" for i in range(size)]
    schema = {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in keys},
        "additionalProperties": False,
    }
    pairs = [(k, i) for i, k in reversed(list(enumerate(keys)))]
    unique = render(pairs, style)
    duplicate = render(pairs + ([(keys[-1], 999)] if keys else [("unknown", 1)]), style)
    compiled, info = compile_object(schema, [unique, duplicate], style, dynamic)
    stop = info.vocab_size - 1
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        for text, token, expected in [(unique, 256, True), (duplicate, 257, False)]:
            check_path(grammar, info, list(text.encode()) + [stop], expected)
            if text:
                check_path(grammar, info, [token, stop], expected)
            else:
                # Empty decoded tokens are special/unusable; the empty XML
                # document itself is accepted above by consuming EOS alone.
                check_path(grammar, info, [token, stop], False)


@pytest.mark.parametrize("dynamic", [False, True])
def test_any_order_distinct_prefixes_unicode_and_scope(dynamic):
    obj = {
        "type": "object",
        "properties": {
            "": {"type": "integer"},
            "a": {"type": "integer"},
            "aa": {"type": "integer"},
            "中": {"type": "integer"},
        },
        "additionalProperties": False,
    }
    schema = {"type": "array", "items": obj}
    good = '[{"aa":2,"a":1,"":0,"中":3},{"a":4}]'
    bad = '[{"a":1,"aa":2,"a":3}]'
    compiled, info = compile_object(schema, [good, bad], dynamic=dynamic)
    for text, token, expected in [(good, 256, True), (bad, 257, False)]:
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [token, info.vocab_size - 1], expected)


def test_any_order_missing_required_cannot_be_replaced_by_optional_keys():
    schema = {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in "abcd"},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    text = '{"c":3,"d":4}'
    compiled, info = compile_object(schema, [text])
    check_path(compiled, info, [256, info.vocab_size - 1], False)


def test_any_order_duplicate_rejection_does_not_commit_state():
    schema = {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in "ab"},
        "additionalProperties": False,
    }
    compiled, info = compile_object(schema, ['{"a":1,', '"a":2}', '"b":2}'])
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(256)
    branch = matcher.fork()
    for _ in range(3):
        assert not matcher.accept_token(257)
    assert matcher.accept_token(258)
    assert matcher.accept_token(info.vocab_size - 1)
    assert matcher.is_terminated()
    branch.rollback(1)
    assert branch.accept_token(256)
    assert branch.accept_token(258)
    assert branch.accept_token(info.vocab_size - 1)


@pytest.mark.parametrize("dynamic", [False, True])
def test_any_order_rule_cache_and_ebnf_roundtrip(dynamic):
    schema = {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in "ab"},
        "additionalProperties": False,
    }
    valid, duplicate = '{"b":2,"a":1}', '{"a":1,"a":2}'
    _, info = compile_object(schema, [valid, duplicate])
    compiler = xgr.GrammarCompiler(
        info, max_threads=8, enable_dynamic_compilation=dynamic, cache_limit_bytes=512 * 1024 * 1024
    )
    # An ordinary schema must not supply context-free accepted masks for guarded keys.
    compiler.compile_json_schema(
        schema, any_order=False, any_whitespace=False, separators=(",", ":")
    )
    raw = xgr.Grammar.from_json_schema(
        schema, any_order=True, any_whitespace=False, separators=(",", ":")
    )
    for source in (raw, str(raw), raw):
        compiled = compiler.compile_grammar(source)
        check_path(compiled, info, [256, info.vocab_size - 1], True)
        check_path(compiled, info, [257, info.vocab_size - 1], False)


@pytest.mark.parametrize(
    "text,expected",
    [
        (r'{"a":1,"\u0061":2}', False),
        ('{"é":1,"\\u00e9":2}', False),
        (r'{"/":1,"\/":2}', False),
        (r'{"\n":1,"\u000a":2}', False),
        (r'{"\\":1,"\u005c":2}', False),
        (r'{"\ud800":1,"\uD800":2}', False),
        ('{"😀":1,"\\ud83d\\ude00":2}', False),
        ('{"é":1,"e\\u0301":2}', True),
        (r'{"\ud800":1,"a":2}', True),
        ('{"x":1,"xx":2}', True),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_any_order_decoded_dynamic_names(text, expected, dynamic):
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    compiled, info = compile_object(schema, [text], dynamic=dynamic)
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        check_path(grammar, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(grammar, info, [256, info.vocab_size - 1], expected)


@pytest.mark.parametrize("style", ["json", "qwen_xml", "minimax_xml"])
def test_any_order_extra_names_and_declared_names_share_identity(style):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": {"type": "integer"},
    }
    texts = [
        render(pairs, style)
        for pairs in [[("x", 1), ("xx", 2), ("a", 3)], [("x", 1), ("x", 2)], [("a", 1), ("a", 2)]]
    ]
    compiled, info = compile_object(schema, texts, style)
    for index, text in enumerate(texts):
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], index == 0)
        check_path(compiled, info, [256 + index, info.vocab_size - 1], index == 0)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "constraints",
    [
        {"pattern": "a{1,65}"},
        {"minLength": 1, "maxLength": 65},
        {"pattern": "a{1,65}", "minLength": 1, "maxLength": 65},
    ],
)
def test_any_order_counted_key_repetition_cannot_bypass_identity(dynamic, constraints):
    schema = {
        "type": "object",
        "propertyNames": {"type": "string", **constraints},
        "additionalProperties": {"type": "integer"},
    }
    good, bad = '{"a":1,"aa":2}', '{"a":1,"a":2}'
    compiled, info = compile_object(schema, [good, bad, 'a"', 'aa"', '"'], dynamic=dynamic)
    for text, token, expected in [(good, 256, True), (bad, 257, False)]:
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [token, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_any_order_complete_small_language_oracle(dynamic):
    # Enumerate the expected language only in this tiny oracle, not in the converter.
    documents = [b"{}", b'{"a":0}', b'{"b":0}', b'{"a":0,"b":0}', b'{"b":0,"a":0}']
    extras = [s.decode() for s in documents] + ['{"a":0,"a":0}', '"a":0}', '0,"a":0}']
    schema = {
        "type": "object",
        "properties": {k: {"const": 0} for k in "ab"},
        "additionalProperties": False,
    }
    _, info = compile_object(schema, extras, dynamic=dynamic)
    # JSONSchemaFormat uses whitespace=true; compact formatting options belong
    # to the direct JSON-schema API, which makes this finite oracle exact.
    compiled = xgr.GrammarCompiler(
        info, max_threads=8, enable_dynamic_compilation=dynamic
    ).compile_json_schema(schema, any_order=True, any_whitespace=False, separators=(",", ":"))
    vocab = [bytes([i]) for i in range(256)] + [s.encode() for s in extras]
    stop = info.vocab_size - 1
    prefixes = sorted({text[:end] for text in documents for end in range(len(text) + 1)})
    for prefix in prefixes:
        matcher, direct = xgr.GrammarMatcher(compiled), xgr.GrammarMatcher(compiled)
        assert matcher.accept_string(prefix.decode()) and direct.accept_string(prefix.decode())
        mask = xgr.allocate_token_bitmask(1, info.vocab_size)
        matcher.fill_next_token_bitmask(mask)
        for token, decoded in enumerate(vocab):
            expected = any(text.startswith(prefix + decoded) for text in documents)
            allowed = bool(int(mask[0, token // 32]) & (1 << (token % 32)))
            assert allowed == direct.fork().accept_token(token) == expected, (prefix, decoded)
        expected_eos = prefix in documents
        assert bool(int(mask[0, stop // 32]) & (1 << (stop % 32))) == expected_eos
        assert direct.fork().accept_token(stop) == expected_eos


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_key_and_value_mask_cache_variants_do_not_alias(dynamic, reverse):
    texts = [r'{"a":1,"\u0061":2}', '{"a":1,"b":2}', '{"v":"a","z":1}']
    key_schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    value_schema = {
        "type": "object",
        "properties": {"v": {"type": "string"}, "z": {"type": "integer"}},
        "additionalProperties": False,
    }
    _, info = compile_object(key_schema, texts)
    compiler = xgr.GrammarCompiler(
        info, max_threads=8, enable_dynamic_compilation=dynamic, cache_limit_bytes=64 * 1024
    )
    schemas = [key_schema, value_schema] if not reverse else [value_schema, key_schema]
    saved = []
    for schema in schemas * 2:
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "style": "json",
                "json_schema": schema,
                "any_order": True,
            },
        }
        compiled = compiler.compile_structural_tag(json.dumps(tag))
        samples = [(0, False), (1, True)] if schema is key_schema else [(2, True)]
        for index, expected in samples:
            check_path(
                compiled, info, list(texts[index].encode()) + [info.vocab_size - 1], expected
            )
            check_path(compiled, info, [256 + index, info.vocab_size - 1], expected)
        saved.append((compiled, samples))
    compiler.clear_cache()
    for compiled, samples in saved:
        restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
        for index, expected in samples:
            check_path(
                restored, info, list(texts[index].encode()) + [info.vocab_size - 1], expected
            )


@pytest.mark.parametrize("codec", ["raw", "json"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_atomic_key_bytes(codec, mixed, dynamic):
    vocab = ["aa", "bb", ":0,", ":0", '"', "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[5])
    leaf = "(Token(0) | Token(1))"
    if mixed:
        leaf = '(Token(0) | "aa" | Token(1) | "bb")'
    if codec == "json":
        leaf = r'"\"" ' + leaf + r' "\""'
    grammar = 'root[object_keys] ::= key ":0," key ":0"\n' + f"key[object_key_{codec}] ::= {leaf}"
    compiled = xgr.GrammarCompiler(
        info, max_threads=8, enable_dynamic_compilation=dynamic
    ).compile_grammar(grammar)
    for second, expected in ((1, True), (0, False)):
        ids = [0, 2, second, 3, 5] if codec == "raw" else [4, 0, 4, 2, 4, second, 4, 3, 5]
        for value in (
            compiled,
            xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
        ):
            check_path(value, info, ids, expected)


@pytest.mark.parametrize(
    "style",
    [
        "json",
        "qwen_xml",
        "minimax_xml",
        "minimax_m3_xml",
        "deepseek_xml",
        "deepseek_v4_1_xml",
        "glm_xml",
        "cohere_xml",
        "kimi_k3_xml",
        "gemma",
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_any_order_all_renderers(style, dynamic):
    schema = {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in "ab"},
        "additionalProperties": False,
    }
    good, bad = render([("b", 2), ("a", 1)], style), render([("a", 1), ("a", 2)], style)
    compiled, info = compile_object(schema, [good, bad], style, dynamic)
    for text, index, expected in [(good, 256, True), (bad, 257, False)]:
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [index, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_any_order_recursive_occurrences_and_union_branches(dynamic):
    node = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "child": {"$ref": "#/$defs/node"}},
        "additionalProperties": False,
    }
    schema = {
        "$defs": {"node": node},
        "type": "array",
        "items": {
            "anyOf": [
                {"$ref": "#/$defs/node"},
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "boolean"}},
                    "additionalProperties": False,
                },
            ]
        },
    }
    good = '[{"a":1,"child":{"a":2}},{"a":"x","b":true},{"child":{"a":3},"a":4}]'
    bad = '[{"a":1,"child":{"a":2,"a":3}}]'
    compiled, info = compile_object(schema, [good, bad], dynamic=dynamic)
    for text, index, expected in [(good, 256, True), (bad, 257, False)]:
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [index, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_completion_before_a_following_duplicate_guard(dynamic, atomic):
    # The second key is mandatory and has already been seen. Completion of the
    # first leaf must be checked even when the token ends exactly at that leaf.
    info = xgr.TokenizerInfo(
        [bytes([i]) for i in range(256)] + [b"aa", b"<eos>"], stop_token_ids=[257]
    )
    body = "Token(256)" if atomic else '"aa"'
    grammar = 'root[object_keys] ::= key key\nkey[object_key="a"] ::= ' + body
    compiled = xgr.GrammarCompiler(
        info, max_threads=8, enable_dynamic_compilation=dynamic
    ).compile_grammar(grammar)
    for value in (compiled, xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)):
        check_path(value, info, [256, 256, 257], False)
        if not atomic:
            check_path(value, info, [97, 97, 97, 97, 257], False)


@pytest.mark.parametrize("dynamic", [False, True])
def test_tail_recursion_preserves_key_identity(dynamic):
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    name = "k" * 1024
    texts = [
        json.dumps({name: "value" * 256, "other": "next"}),
        '{"' + name + '":"x","' + name + '":"y"}',
    ]
    compiled, info = compile_object(schema, texts, dynamic=dynamic)
    for text, index, expected in [(texts[0], 256, True), (texts[1], 257, False)]:
        check_path(compiled, info, [index, info.vocab_size - 1], expected)
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_completion_before_nullable_dynamic_key_guard(dynamic, atomic):
    info = xgr.TokenizerInfo(
        [bytes([i]) for i in range(256)] + [b"aa", b"<eos>"], stop_token_ids=[257]
    )
    body = "Token(256)" if atomic else '"aa"'
    compiler = xgr.GrammarCompiler(info, max_threads=8, enable_dynamic_compilation=dynamic)
    for repeat, expected in [(False, True), (True, False)]:
        grammar = (
            "root[object_keys] ::= empty_key leaf"
            + (" empty_key" if repeat else "")
            + '\nempty_key[object_key_raw] ::= ""\nleaf[capture="body"] ::= '
            + body
        )
        compiled = compiler.compile_grammar(grammar)
        for value in (
            compiled,
            xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
        ):
            check_path(value, info, [256, 257], expected)
            if not atomic:
                check_path(value, info, [97, 97, 257], expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("inner,expected", [("a", False), ("b", True)])
def test_declared_key_completion_preserves_inner_key_updates(dynamic, inner, expected):
    info = xgr.TokenizerInfo(
        [bytes([i]) for i in range(256)] + [b"xyz", b"<eos>"], stop_token_ids=[257]
    )
    grammar = (
        'root[object_keys] ::= outer\nouter[object_key="a"] ::= "x" inner "z"\n'
        f'inner[object_key="{inner}"] ::= "y"'
    )
    compiled = xgr.GrammarCompiler(
        info, max_threads=8, enable_dynamic_compilation=dynamic
    ).compile_grammar(grammar)
    for value in (compiled, xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)):
        check_path(value, info, [256, 257], expected)
        check_path(value, info, list(b"xyz") + [257], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_long_json_key_switches_to_escape_decoder_without_changing_identity(dynamic):
    prefix = "x" * 128
    first = json.dumps(prefix + "é", ensure_ascii=False)
    alias = '"' + prefix + r'\u00e9"'
    good = "{" + first + ':1,"different":2}'
    bad = "{" + first + ":1," + alias + ":2}"
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    compiled, info = compile_object(schema, [good, bad], dynamic=dynamic)
    for text, index, expected in [(good, 256, True), (bad, 257, False)]:
        check_path(compiled, info, [index, info.vocab_size - 1], expected)
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
