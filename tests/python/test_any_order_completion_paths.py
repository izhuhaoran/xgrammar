# Copyright (c) 2026 by Contributors
"""Black-box key completion tests across shared and captured rule boundaries."""

import json

import pytest

import xgrammar as xgr

TOKENIZATIONS = ("bytes", "leaf_macro", "document_macro")
ALIAS_CASES = (
    # The second name is a complete JSON key lexeme, including its quotes.
    ("json", "é", r'"\u00e9"', False),
    ("json", "é", r'"\u00ea"', True),
    ("raw", "é", r'"\u00e9"', False),
    ("raw", "é", r'"\u00ea"', True),
    ("json", "a/b", r'"a\/b"', False),
    ("json", "a/b", r'"a\/c"', True),
)


def _literal(text):
    # EBNF string literals use the same escapes needed by these small fixtures.
    return json.dumps(text, ensure_ascii=False)


def _token_ids(raw, leaf, tokenization, leaf_id, document_id, eos_id):
    if tokenization == "bytes":
        return list(raw) + [eos_id]
    if tokenization == "document_macro":
        return [document_id, eos_id]
    assert tokenization == "leaf_macro"
    assert leaf and leaf in raw
    ids = []
    offset = 0
    while True:
        start = raw.find(leaf, offset)
        if start < 0:
            ids.extend(raw[offset:])
            break
        ids.extend(raw[offset:start])
        ids.append(leaf_id)
        offset = start + len(leaf)
    return ids + [eos_id]


def _compile(source, text, leaf_text, tokenization, dynamic, serde):
    raw = text.encode("utf-8")
    leaf = leaf_text.encode("utf-8")
    vocab = [bytes([value]) for value in range(256)] + [leaf, raw, b"<eos>"]
    leaf_id, document_id, eos_id = 256, 257, 258
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[eos_id])
    grammar = xgr.Grammar.from_ebnf(source)
    if serde:
        grammar = xgr.Grammar.deserialize_json(grammar.serialize_json())
    compiler = xgr.GrammarCompiler(info, max_threads=8, enable_dynamic_compilation=dynamic)
    compiled = compiler.compile_grammar(grammar)
    if serde:
        compiled = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
    ids = _token_ids(raw, leaf, tokenization, leaf_id, document_id, eos_id)
    return compiled, info, ids


def _check_path(compiled, info, ids, expected, captures=None):
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=False)
    direct = xgr.GrammarMatcher(compiled, terminate_without_stop_token=False)
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    eos_id = info.stop_token_ids[0]
    rejected = False
    for step, token_id in enumerate(ids):
        if expected and token_id == eos_id:
            assert matcher.is_completed()
            assert direct.is_completed()
            assert not matcher.is_terminated()
            assert not direct.is_terminated()
        xgr.reset_token_bitmask(mask)
        matcher.fill_next_token_bitmask(mask)
        word = int(mask[0, token_id // 32]) & 0xFFFFFFFF
        allowed = bool(word & (1 << (token_id % 32)))
        accepted_direct = bool(direct.accept_token(token_id))
        accepted_masked = bool(matcher.accept_token(token_id))
        assert allowed == accepted_direct, (step, token_id)
        assert accepted_masked == accepted_direct, (step, token_id)
        if not accepted_direct:
            rejected = True
            break
    if expected:
        assert not rejected
        assert matcher.is_terminated()
        assert direct.is_terminated()
        assert matcher.get_captures() == direct.get_captures()
        if captures is not None:
            assert matcher.get_captures() == captures
    else:
        assert rejected
        assert not matcher.is_terminated()
        assert not direct.is_terminated()


@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("serde", [False, True], ids=["native", "serde"])
@pytest.mark.parametrize("second_key,expected", [("a", False), ("b", True)])
def test_capture_completion_preserves_following_key_guard(
    tokenization, dynamic, serde, second_key, expected
):
    source = (
        "root[object_keys] ::= first leaf wrapped\n"
        'first[object_key="a"] ::= "x"\n'
        'leaf[capture="body"] ::= "aa"\n'
        'wrapped[capture="tail"] ::= second "z"\n'
        f'second[object_key={_literal(second_key)}] ::= "y"\n'
    )
    compiled, info, ids = _compile(source, "xaayz", "aa", tokenization, dynamic, serde)
    _check_path(compiled, info, ids, expected, captures=[("body", b"aa"), ("tail", b"yz")])


@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("serde", [False, True], ids=["native", "serde"])
@pytest.mark.parametrize("codec", ["raw", "json"])
@pytest.mark.parametrize("extended", [False, True], ids=["duplicate_whole", "distinct_extension"])
def test_shared_key_leaf_completion_can_continue_to_a_distinct_name(
    tokenization, dynamic, serde, codec, extended
):
    # The same captured leaf finishes the whole short raw key but is only the
    # front of the longer key. For JSON it is inside both quoted key lexemes.
    # Seeing "aa" in the first key must still permit the second key "aab".
    if codec == "raw":
        whole = "shared"
        longer = 'shared "b"'
        text = "aa|" + ("aab" if extended else "aa")
    else:
        whole = r'"\"" shared "\""'
        longer = r'"\"" shared "b" "\""'
        text = '"aa"|' + ('"aab"' if extended else '"aa"')
    source = (
        'root[object_keys] ::= whole "|" (whole | extended)\n'
        f"whole[object_key_{codec}] ::= {whole}\n"
        f"extended[object_key_{codec}] ::= {longer}\n"
        'shared[capture="stem"] ::= "aa"\n'
    )
    compiled, info, ids = _compile(source, text, "aa", tokenization, dynamic, serde)
    _check_path(compiled, info, ids, extended, captures=[("stem", b"aa"), ("stem", b"aa")])


@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("serde", [False, True], ids=["native", "serde"])
@pytest.mark.parametrize("first_codec,first_name,second_lexeme,expected", ALIAS_CASES)
def test_unicode_and_escaped_key_aliases_across_completion_wrappers(
    tokenization, dynamic, serde, first_codec, first_name, second_lexeme, expected
):
    # Only valid, previously supported Unicode and JSON escape spellings are
    # used. Identity follows JSON decoding; this adds no normalization policy.
    assert expected == (first_name != json.loads(second_lexeme))
    if first_codec == "json":
        first_body = r'"\"" head "\""'
        first_text = json.dumps(first_name, ensure_ascii=False)
    else:
        first_body = "head"
        first_text = first_name
    source = (
        'root[object_keys] ::= first "|" second\n'
        f"first[object_key_{first_codec}] ::= {first_body}\n"
        f'head[capture="head"] ::= {_literal(first_name)}\n'
        f"second[object_key_json] ::= {_literal(second_lexeme)}\n"
    )
    compiled, info, ids = _compile(
        source, first_text + "|" + second_lexeme, first_name, tokenization, dynamic, serde
    )
    _check_path(compiled, info, ids, expected, captures=[("head", first_name.encode("utf-8"))])


@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize("dynamic", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("serde", [False, True], ids=["native", "serde"])
@pytest.mark.parametrize("nested", [False, True])
def test_shared_value_completion_cannot_bypass_a_completed_key_guard(
    tokenization, dynamic, serde, nested
):
    # Temperature keeps the value rule separate without capture metadata, which
    # would independently disable the tail-call optimization under test.
    source = (
        'object[object_keys] ::= "{" item "," item "}"\n'
        'item ::= named | other | dynamic\n'
        'named[object_key="a"] ::= "\\\"a\\\":" value\n'
        'other[object_key="b"] ::= "\\\"b\\\":" value\n'
        'dynamic ::= key ":" value\n'
        'key[object_key_json] ::= "\\\"" [ab] "\\\""\n'
        'value[temperature=0.5] ::= [0-9]\n'
    )
    source += (
        'root[object_keys] ::= outer\nouter[object_key="payload"] ::= "p" object "q"\n'
        if nested
        else "root ::= object\n"
    )
    for body, expected in (
        ('{"a":1,"b":2}', True),
        ('{"b":1,"a":2}', True),
        ('{"a":1,"a":2}', False),
        ('{"a":1,"a":1}', False),
    ):
        text = "p" + body + "q" if nested else body
        compiled, info, ids = _compile(source, text, "1", tokenization, dynamic, serde)
        _check_path(compiled, info, ids, expected)
