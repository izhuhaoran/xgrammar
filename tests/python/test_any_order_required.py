# Copyright (c) 2026 by Contributors
"""Required-identity contracts for any-order objects.

Closed objects must satisfy required by logical key identity. Existing
additional/unevaluated/strict parsing decides whether an object is closed.
Explicitly open objects retain the existing count-only required lower bound.
Unknown required names do not add compile errors. The pre-existing rejection
of maxProperties below the required count remains an explicit baseline control.

The matrix is deliberately factored: large identity widths use JSON/Qwen XML,
renderer breadth uses a tiny fixed schema, and roundtrips/stateful wrappers use
small inputs. All vocabularies are synthetic byte tokens plus short macro
pieces. There are no per-byte forks or full-vocabulary dependencies.
"""

import itertools
import json

import pytest
from test_any_order_keys import check_path, compile_object, render

import xgrammar as xgr

CORE_STYLES = ("json", "qwen_xml")
STYLES = (
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
)
REPRESENTATIONS = ("native", "grammar_serde", "compiled_serde", "ebnf")


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _closed(properties, required, **extra):
    return {
        "type": "object",
        "properties": {name: {"type": "integer"} for name in properties},
        "required": list(required),
        "additionalProperties": False,
        **extra,
    }


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("key, encoded", [("a&b", "a&amp;b"), ("a&amp;b", "a&amp;amp;b")])
def test_cohere_required_uses_the_same_encoded_identity(key, encoded, dynamic):
    schema = _closed([key, "z"], [key])
    good = f'<cofl:value name="{encoded}" type="json">1</cofl:value>'
    bad = '<cofl:value name="z" type="json">2</cofl:value>'
    compiled, info = compile_object(schema, [good, bad], "cohere_xml", dynamic)
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        for text, token, expected in ((good, 256, True), (bad, 257, False)):
            check_path(grammar, info, list(text.encode()) + [info.vocab_size - 1], expected)
            check_path(grammar, info, [token, info.vocab_size - 1], expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_unknown_cohere_required_name_does_not_add_key_validation(dynamic):
    # A legal but unsatisfiable schema is not a new compile-time key error.
    text = '<cofl:value name="x" type="json">1</cofl:value>'
    compiled, info = compile_object(_closed(["x"], ["\x00"]), [text], "cohere_xml", dynamic)
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        check_path(grammar, info, list(text.encode()) + [info.vocab_size - 1], False)
        check_path(grammar, info, [256, info.vocab_size - 1], False)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "key, encoded, other_encoded",
    [
        ("a&b", "a&amp;b", "a&amp;amp;b"),
        ('a"b', "a&quot;b", "a&amp;quot;b"),
        ("a&amp;b", "a&amp;amp;b", "a&amp;b"),
    ],
)
def test_kimi_unknown_pattern_required_uses_encoded_identity(key, encoded, other_encoded, dynamic):
    schema = {
        "type": "object",
        "properties": {"spare": {"type": "integer"}},
        "patternProperties": {"^(" + encoded + "|" + other_encoded + ")$": {"type": "integer"}},
        "required": [key],
        "additionalProperties": False,
    }
    # The required name is undeclared and can only arrive through the raw pattern.
    # Another permitted encoded name satisfies the entry count, not this identity.
    cases = [
        ([(encoded, 1)], True),
        ([("spare", 1)], False),
        ([(other_encoded, 1)], False),
        ([(other_encoded, 1), (encoded, 2)], True),
    ]
    texts = [render(pairs, "kimi_k3_xml") for pairs, _ in cases]
    compiled, info = compile_object(schema, texts, "kimi_k3_xml", dynamic)
    for index, (_, expected) in enumerate(cases):
        text = texts[index]
        check_path(compiled, info, list(text.encode()) + [info.vocab_size - 1], expected)
        check_path(compiled, info, [256 + index, info.vocab_size - 1], expected)


def _lexicon(texts):
    result = []
    for text in texts:
        pieces = [text]
        if len(text) > 1:
            cut = len(text) // 2
            pieces.extend((text[:cut], text[cut:]))
        for piece in pieces:
            if piece and piece not in result:
                result.append(piece)
    return result


def _info(texts):
    vocab = [bytes([value]) for value in range(256)]
    vocab.extend(text.encode("utf-8") for text in _lexicon(texts))
    vocab.append(b"<eos>")
    return xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])


def _macro_id(info, text):
    raw = text.encode("utf-8")
    for token_id, token in enumerate(info.decoded_vocab[256:], start=256):
        if token == raw:
            return token_id
    raise AssertionError(f"fixture macro is absent: {text!r}")


def _plans(info, text):
    eos = info.stop_token_ids[0]
    raw = text.encode("utf-8")
    plans = [list(raw) + [eos]]
    if text:
        plans.append([_macro_id(info, text), eos])
    if len(text) > 1:
        cut = len(text) // 2
        plans.append([_macro_id(info, text[:cut]), _macro_id(info, text[cut:]), eos])
    # Empty XML content has only EOS, not an ordinary empty token.
    return [list(plan) for plan in dict.fromkeys(tuple(plan) for plan in plans)]


def _check_cases(compiled, info, cases):
    for text, expected in cases:
        for ids in _plans(info, text):
            check_path(compiled, info, ids, expected)


def _compiler(info, dynamic):
    return xgr.GrammarCompiler(info, max_threads=8, enable_dynamic_compilation=dynamic)


def _restore_compiled(compiled, info, representation):
    assert representation in ("native", "compiled_serde")
    if representation == "compiled_serde":
        # Public dynamic serialization materializes the eager wire format.
        # This checks required semantics after restore, not a lazy-state format.
        return xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
    return compiled


def _compile_renderer(schema, cases, style, dynamic, representation="native"):
    texts = _lexicon([text for text, _ in cases])
    compiled, info = compile_object(schema, texts, style=style, dynamic=dynamic)
    return _restore_compiled(compiled, info, representation), info


def _compile_json(schema, cases, dynamic, strict=True, representation="native"):
    info = _info([text for text, _ in cases])
    compiler = _compiler(info, dynamic)
    kwargs = dict(any_order=True, any_whitespace=False, separators=(",", ":"), strict_mode=strict)
    if representation in ("native", "compiled_serde"):
        compiled = compiler.compile_json_schema(schema, **kwargs)
        return _restore_compiled(compiled, info, representation), info
    grammar = xgr.Grammar.from_json_schema(schema, **kwargs)
    if representation == "grammar_serde":
        grammar = xgr.Grammar.deserialize_json(grammar.serialize_json())
    else:
        assert representation == "ebnf"
        # No knowledge of the new required annotation spelling is needed.
        grammar = xgr.Grammar.from_ebnf(str(grammar))
    return compiler.compile_grammar(grammar), info


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", CORE_STYLES)
def test_closed_required_small_unique_language_oracle(style, dynamic):
    names = ("a", "b", "spare")
    required = ("a", "b")
    schema = _closed(names, required, maxProperties=3)
    cases = []
    for count in range(4):
        for order in itertools.permutations(names, count):
            # Unique keys and integer values are valid independently of xgr.
            expected = set(required).issubset(order)
            cases.append((render([(name, 0) for name in order], style), expected))
    compiled, info = _compile_renderer(schema, cases, style, dynamic)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("closure", ["strict_default", "explicit_false", "unevaluated_false"])
def test_required_identity_follows_existing_effective_closure(closure, dynamic):
    schema = {
        "type": "object",
        "properties": {"need": {"type": "integer"}, "spare": {"type": "integer"}},
        "required": ["need"],
    }
    strict = closure == "strict_default"
    if closure == "explicit_false":
        schema["additionalProperties"] = False
    elif closure == "unevaluated_false":
        schema["unevaluatedProperties"] = False
    # In this baseline, non-strict additional properties remain enabled even
    # with unevaluatedProperties=false. Keep that existing effective-open mode.
    effective_open = closure == "unevaluated_false"
    cases = [
        ("{}", False),
        ('{"need":0}', True),
        ('{"spare":0}', effective_open),
        ('{"spare":0,"need":1}', True),
        ('{"need":0,"unknown":1}', effective_open),
    ]
    compiled, info = _compile_json(schema, cases, dynamic, strict=strict)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("openness", ["true", "schema", "nonstrict_default"])
def test_open_objects_keep_existing_required_count_only(openness, dynamic):
    schema = {
        "type": "object",
        "properties": {
            "need_a": {"type": "integer"},
            "need_b": {"type": "integer"},
            "spare": {"type": "integer"},
        },
        "required": ["need_a", "need_b"],
        "maxProperties": 3,
    }
    if openness != "nonstrict_default":
        schema["additionalProperties"] = True if openness == "true" else {"type": "integer"}
    cases = [
        ("{}", False),
        ('{"other":0}', False),
        ('{"spare":0,"other":1}', True),  # Two entries, neither required identity.
        ('{"need_a":0,"other":1}', True),  # One required identity is absent.
        ('{"need_a":0,"need_b":1}', True),
    ]
    compiled, info = _compile_json(schema, cases, dynamic, strict=openness != "nonstrict_default")
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("openness", ["true", "schema"])
def test_open_qwen_xml_keeps_existing_required_count_only(openness, dynamic):
    schema = {
        "type": "object",
        "properties": {
            "need_a": {"type": "integer"},
            "need_b": {"type": "integer"},
            "spare": {"type": "integer"},
        },
        "required": ["need_a", "need_b"],
        "additionalProperties": True if openness == "true" else {"type": "integer"},
        "maxProperties": 3,
    }
    cases = [
        (render([], "qwen_xml"), False),
        (render([("other", 0)], "qwen_xml"), False),
        (render([("spare", 0), ("other", 1)], "qwen_xml"), True),
        (render([("need_a", 0), ("other", 1)], "qwen_xml"), True),
        (render([("need_a", 0), ("need_b", 1)], "qwen_xml"), True),
    ]
    compiled, info = _compile_renderer(schema, cases, "qwen_xml", dynamic)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", STYLES)
def test_required_identity_all_existing_renderers(style, dynamic):
    schema = _closed(("a", "b", "spare"), ("a", "b"), maxProperties=3)
    cases = [
        (render([("b", 1), ("a", 0)], style), True),
        (render([("a", 0), ("spare", 1)], style), False),
        (render([("spare", 2), ("b", 1), ("a", 0)], style), True),
    ]
    compiled, info = _compile_renderer(schema, cases, style, dynamic)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", CORE_STYLES)
@pytest.mark.parametrize("required_count", [0, 1, 64, 65, 128])
def test_required_identity_width_boundaries(required_count, style, dynamic):
    required = [f"need_{index:03d}" for index in range(required_count)]
    properties = required + (["spare"] if required_count else [])
    schema = _closed(properties, required, maxProperties=len(properties))
    full = [(name, index) for index, name in reversed(list(enumerate(required)))]
    cases = [(render(full, style), True)]
    if required_count:
        missing = [(name, value) for name, value in full if name != required[-1]]
        missing.append(("spare", 999))
        assert len(missing) == required_count
        cases.extend(
            ((render(missing, style), False), (render(full + [("spare", 999)], style), True))
        )
    else:
        cases.append((render([("spare", 0)], style), False))
    compiled, info = _compile_renderer(schema, cases, style, dynamic)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", CORE_STYLES)
@pytest.mark.parametrize("declared", [False, True], ids=["pattern_only", "mixed_properties"])
def test_required_name_can_be_undeclared_but_allowed_by_pattern(style, dynamic, declared):
    schema = {
        "type": "object",
        "patternProperties": {"^p[01]$": {"type": "integer"}},
        "required": ["p0"],
        "additionalProperties": False,
        "maxProperties": 3,
    }
    prefix = []
    if declared:
        schema["properties"] = {"known": {"type": "integer"}}
        schema["required"].append("known")
        prefix = [("known", 9)]
    # The pattern-only form intentionally has no properties member.
    cases = [
        (render(prefix + [("p0", 0)], style), True),
        (render(prefix + [("p1", 0)], style), False),
        (render(prefix + [("p1", 1), ("p0", 0)], style), True),
        (render(prefix, style), False),
    ]
    compiled, info = _compile_renderer(schema, cases, style, dynamic)
    _check_cases(compiled, info, cases)


def _edge_case(kind, style):
    if kind == "unknown_required":
        schema = _closed(("a",), ("missing",))
        rows = [([], False), ([("a", 0)], False), ([("missing", 0)], False)]
    elif kind == "pattern_unknown_required":
        schema = {
            "type": "object",
            "patternProperties": {"^p[01]$": {"type": "integer"}},
            "required": ["missing"],
            "additionalProperties": False,
        }
        rows = [([], False), ([("p0", 0)], False), ([("missing", 0)], False)]
    elif kind == "no_properties_required":
        schema = {"type": "object", "required": ["missing"], "additionalProperties": False}
        rows = [([], False), ([("missing", 0)], False)]
    elif kind == "zero_max_unsatisfiable":
        schema = _closed(("a",), ("a",), maxProperties=0)
        rows = [([], False), ([("a", 0)], False)]
    else:
        assert kind == "zero_max_empty_required"
        schema = _closed(("a",), (), maxProperties=0)
        rows = [([], True), ([("a", 0)], False)]
    return schema, [(render(pairs, style), expected) for pairs, expected in rows]


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", CORE_STYLES)
@pytest.mark.parametrize(
    "kind",
    [
        "unknown_required",
        "pattern_unknown_required",
        "no_properties_required",
        "zero_max_unsatisfiable",
        "zero_max_empty_required",
    ],
)
def test_closed_edge_schemas_preserve_existing_compile_boundary(kind, style, dynamic):
    schema, cases = _edge_case(kind, style)
    if kind == "zero_max_unsatisfiable":
        # Valid but unsatisfiable JSON Schema: main already rejects this count
        # conflict before object conversion. This change does not remove it.
        with pytest.raises(
            RuntimeError, match="maxProperties is less than the number of required properties"
        ):
            _compile_renderer(schema, cases, style, dynamic)
        return
    # Unknown required names may be supplied by a pattern or be unsatisfiable;
    # neither case introduces a defensive schema-validation error.
    compiled, info = _compile_renderer(schema, cases, style, dynamic)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("schema_kind", ["declared", "pattern_only"])
def test_required_metadata_public_json_roundtrips(dynamic, representation, schema_kind):
    if schema_kind == "declared":
        schema = _closed(("a", "b", "x"), ("a", "b"), maxProperties=3)
    else:
        schema = {
            "type": "object",
            "patternProperties": {"^[abx]$": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
            "maxProperties": 3,
        }
    cases = [
        ('{"b":1,"a":0}', True),
        ('{"a":0,"x":1}', False),
        ('{"b":1,"x":2}', False),
        ('{"x":2,"b":1,"a":0}', True),
    ]
    compiled, info = _compile_json(schema, cases, dynamic, representation=representation)
    _check_cases(compiled, info, cases)


def _nested_cases(kind):
    item = _closed(("need", "spare"), ("need",))
    if kind == "array_refs":
        schema = {
            "$defs": {"item": item},
            "type": "array",
            "items": {"$ref": "#/$defs/item"},
            "minItems": 2,
            "maxItems": 2,
        }
        values = [
            ([{"need": 0}, {"need": 1}], True),
            ([{"need": 0}, {"spare": 1}], False),
            ([{"spare": 0}, {"need": 1}], False),
            ([{"spare": 9, "need": 0}, {"need": 1}], True),
        ]
    elif kind == "nested_same_name":
        schema = {
            "$defs": {"item": item},
            "type": "object",
            "properties": {"need": {"$ref": "#/$defs/item"}, "other": {"$ref": "#/$defs/item"}},
            "required": ["need"],
            "additionalProperties": False,
        }
        values = [
            ({"need": {"need": 0}}, True),
            ({"need": {"spare": 0}}, False),
            ({"other": {"need": 0}}, False),
            ({"other": {"need": 1}, "need": {"need": 0}}, True),
        ]
    else:
        assert kind == "recursive_ref"
        node = {
            "type": "object",
            "properties": {"need": {"type": "integer"}, "child": {"$ref": "#/$defs/node"}},
            "required": ["need"],
            "additionalProperties": False,
        }
        schema = {"$defs": {"node": node}, "$ref": "#/$defs/node"}
        values = [
            ({"need": 0, "child": {"need": 1}}, True),
            ({"need": 0, "child": {"child": {"need": 2}}}, False),
            ({"child": {"need": 1}}, False),
            ({"need": 0, "child": {"need": 1, "child": {"need": 2}}}, True),
        ]
    return schema, [(_dump(value), expected) for value, expected in values]


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("representation", ["native", "ebnf"])
@pytest.mark.parametrize("kind", ["array_refs", "nested_same_name", "recursive_ref"])
def test_required_is_local_to_nested_array_and_ref_occurrences(kind, representation, dynamic):
    schema, cases = _nested_cases(kind)
    compiled, info = _compile_json(schema, cases, dynamic, representation=representation)
    _check_cases(compiled, info, cases)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", CORE_STYLES)
@pytest.mark.parametrize("representation", ["native", "compiled_serde"])
def test_required_resets_between_shared_named_grammar_occurrences(style, dynamic, representation):
    schema = _closed(("need", "spare"), ("need",))
    good = render([("need", 0)], style)
    missing = render([("spare", 0)], style)
    cases = [
        (good + "|" + good, True),
        (good + "|" + missing, False),
        (missing + "|" + good, False),
    ]
    base, info = compile_object(schema, _lexicon([text for text, _ in cases]), style, dynamic)
    compiled = _compiler(info, dynamic).compile_lark(
        'start: @item "|" @item', named_grammars={"item": base.grammar}
    )
    compiled = _restore_compiled(compiled, info, representation)
    _check_cases(compiled, info, cases)


def _pair(compiled):
    return (xgr.GrammarMatcher(compiled), xgr.GrammarMatcher(compiled))


def _step(pair, info, token_id, expected=None):
    matcher, reference = pair
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    matcher.fill_next_token_bitmask(mask)
    allowed = bool((int(mask[0, token_id // 32]) & 0xFFFFFFFF) & (1 << (token_id % 32)))
    direct = bool(reference.accept_token(token_id))
    actual = bool(matcher.accept_token(token_id))
    assert allowed == direct, token_id
    assert actual == direct, token_id
    if expected is not None:
        assert actual == expected, token_id
    return actual


def _finish(pair, info):
    assert all(matcher.is_completed() for matcher in pair)
    assert all(not matcher.is_terminated() for matcher in pair)
    _step(pair, info, info.stop_token_ids[0], True)
    assert all(matcher.is_terminated() for matcher in pair)


def _walk_pair(compiled, info, ids, expected):
    pair = _pair(compiled)
    rejected = False
    for token_id in ids:
        if expected and token_id == info.stop_token_ids[0]:
            assert all(matcher.is_completed() for matcher in pair)
        if not _step(pair, info, token_id):
            rejected = True
            break
    if expected:
        assert not rejected
        assert all(matcher.is_terminated() for matcher in pair)
    else:
        assert rejected
        assert all(not matcher.is_terminated() for matcher in pair)
    return pair


def _tagged_compile(schema, texts, style, dynamic):
    info = _info(texts)
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "<o>",
            "content": {
                "type": "json_schema",
                "json_schema": schema,
                "style": style,
                "any_order": True,
            },
            "end": "</o>",
        },
    }
    return _compiler(info, dynamic).compile_structural_tag(json.dumps(tag)), info


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("style", CORE_STYLES)
def test_required_fork_rollback_and_rejected_close_retry(style, dynamic):
    schema = _closed(("a", "b", "spare"), ("a", "b"), maxProperties=3)
    first = render([("a", 0)], style)
    if style == "json":
        prefix = "<o>" + first[:-1]
        add_b = "," + render([("b", 1)], style)[1:-1]
        add_spare = "," + render([("spare", 2)], style)[1:-1]
        close = "}</o>"
    else:
        prefix = "<o>" + first
        add_b = render([("b", 1)], style)
        add_spare = render([("spare", 2)], style)
        close = "</o>"
    cases = [
        (prefix + add_b + close, True),
        (prefix + add_spare + close, False),
        (prefix + add_spare + add_b + close, True),
    ]
    compiled, info = _tagged_compile(
        schema, [text for text, _ in cases] + [prefix, add_b, add_spare, close], style, dynamic
    )
    _check_cases(compiled, info, cases)
    p, b, spare, end = [_macro_id(info, text) for text in (prefix, add_b, add_spare, close)]

    original = _pair(compiled)
    _step(original, info, p, True)
    _step(original, info, spare, True)
    forked = tuple(matcher.fork() for matcher in original)
    _step(forked, info, end, False)
    _step(forked, info, end, False)  # Rejection must leave a retryable prefix.
    _step(forked, info, b, True)
    _step(forked, info, end, True)
    _finish(forked, info)
    # Satisfying b on the fork must not satisfy b on its parent occurrence.
    _step(original, info, end, False)
    _step(original, info, b, True)
    _step(original, info, end, True)
    _finish(original, info)

    rolled_back = _pair(compiled)
    _step(rolled_back, info, p, True)
    _step(rolled_back, info, b, True)
    for matcher in rolled_back:
        matcher.rollback(1)
    _step(rolled_back, info, spare, True)
    _step(rolled_back, info, end, False)
    _step(rolled_back, info, b, True)
    _step(rolled_back, info, end, True)
    _finish(rolled_back, info)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("representation", ["native", "compiled_serde"])
def test_required_composes_with_capture_atomic_tokens_and_occurrence_budgets(
    dynamic, representation
):
    schema = _closed(("need", "spare"), ("need",), maxProperties=2)
    good, missing = '{"need":0}', '{"spare":0}'
    opening, closing = "<p>", "</p>"
    whole = opening + good + "|" + good + closing
    info = _info([opening, closing, good, missing, whole])
    opening_id, closing_id = _macro_id(info, opening), _macro_id(info, closing)
    arguments = xgr.Grammar.from_json_schema(
        schema, any_order=True, any_whitespace=False, separators=(",", ":")
    )
    grammar = (
        f'start: <[{opening_id}]> item "|" item <[{closing_id}]>\n'
        'item[capture="document", max_tokens=64]: @arguments\n'
    )
    compiled = _compiler(info, dynamic).compile_lark(
        grammar, named_grammars={"arguments": arguments}
    )
    compiled = _restore_compiled(compiled, info, representation)
    for first, second, expected in (
        (good, good, True),
        (good, missing, False),
        (missing, good, False),
    ):
        for payload_path in ("macro", "bytes"):
            if payload_path == "macro":
                first_ids = [_macro_id(info, first)]
                second_ids = [_macro_id(info, second)]
            else:
                first_ids = list(first.encode())
                second_ids = list(second.encode())
            ids = (
                [opening_id]
                + first_ids
                + [ord("|")]
                + second_ids
                + [closing_id, info.stop_token_ids[0]]
            )
            pair = _walk_pair(compiled, info, ids, expected)
            if expected:
                wanted = [("document", good.encode()), ("document", good.encode())]
                assert pair[0].get_captures() == wanted
                assert pair[1].get_captures() == wanted
    # Marker atoms match their token IDs, not byte-equivalent spellings or a
    # different ordinary token containing the whole protocol message.
    check_path(compiled, info, [ord("<"), info.stop_token_ids[0]], False)
    check_path(compiled, info, [_macro_id(info, whole), info.stop_token_ids[0]], False)


@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("representation", ["native", "compiled_serde"])
def test_required_after_an_exact_one_token_budget_gate(dynamic, representation):
    schema = _closed(("need", "spare"), ("need",), maxProperties=2)
    good, missing = '{"need":0}', '{"spare":0}'
    opening, closing = "<p>", "</p>"
    info = _info([opening, closing, "aa", good, missing])
    opening_id, closing_id = _macro_id(info, opening), _macro_id(info, closing)
    arguments = xgr.Grammar.from_json_schema(
        schema, any_order=True, any_whitespace=False, separators=(",", ":")
    )
    grammar = (
        f'start: <[{opening_id}]> gate "|" item <[{closing_id}]>\n'
        'gate[capture="gate", max_tokens=1]: /a+/\n'
        'item[capture="document", max_tokens=64]: @arguments\n'
    )
    compiled = _compiler(info, dynamic).compile_lark(
        grammar, named_grammars={"arguments": arguments}
    )
    compiled = _restore_compiled(compiled, info, representation)
    # The gate may end after every a, so its one-token limit is exact. This
    # avoids making assumptions about relaxation inside an incomplete object.
    paths = [
        ([_macro_id(info, "aa")], b"aa", good, True),
        ([ord("a")], b"a", good, True),
        ([ord("a"), ord("a")], b"aa", good, False),
        ([_macro_id(info, "aa")], b"aa", missing, False),
    ]
    for gate_ids, gate_capture, payload, expected in paths:
        ids = (
            [opening_id]
            + gate_ids
            + [ord("|"), _macro_id(info, payload), closing_id, info.stop_token_ids[0]]
        )
        if len(gate_ids) > 1:
            # max_tokens is enforced during mask-driven generation (EBNF API
            # contract). A fresh accept-only matcher has no pending mask policy.
            matcher = xgr.GrammarMatcher(compiled)
            mask = xgr.allocate_token_bitmask(1, info.vocab_size)
            rejected = False
            for token_id in ids:
                matcher.fill_next_token_bitmask(mask)
                allowed = bool(int(mask[0, token_id // 32]) & (1 << (token_id % 32)))
                accepted = matcher.accept_token(token_id)
                assert allowed == accepted
                if not accepted:
                    rejected = True
                    break
            assert not expected and rejected
            continue
        pair = _walk_pair(compiled, info, ids, expected)
        if expected:
            wanted = [("gate", gate_capture), ("document", good.encode())]
            assert pair[0].get_captures() == wanted
            assert pair[1].get_captures() == wanted
