# Copyright (c) 2026 by Contributors
"""Public atomic/byte continuations across any-order object occurrences."""

import json
import re

import pytest

import xgrammar as xgr


def _object(name):
    return {
        "type": "object",
        "properties": {name: {"type": "integer"}},
        "required": [name],
        "additionalProperties": False,
    }


def _info(prefix, suffixes):
    vocab = [prefix] + [bytes([value]) for value in range(256)]
    vocab.extend(suffix.encode() for suffix in suffixes)
    vocab.append(b"<eos>")
    return xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])


def _walk(compiled, info, prefix, suffix, expected, action="plain"):
    direct, masked = xgr.GrammarMatcher(compiled), xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)

    def advance(token):
        masked.fill_next_token_bitmask(mask)
        allowed = bool(int(mask[0, token // 32]) & (1 << (token % 32)))
        accepted = direct.accept_token(token)
        assert masked.accept_token(token) == accepted
        assert allowed == accepted, masked._debug_print_internal_state()
        return accepted

    for token in prefix:
        assert advance(token)
    if action == "fork":
        direct, masked = direct.fork(), masked.fork()
    elif action == "rollback":
        direct.rollback(len(prefix))
        masked.rollback(len(prefix))
        for token in prefix:
            assert advance(token)
    for token in suffix + [info.stop_token_ids[0]]:
        if not advance(token):
            assert not expected, direct._debug_print_internal_state()
            return
    assert expected and direct.is_terminated() and masked.is_terminated()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("unused_object", [False, True])
def test_mixed_dispatch_keeps_ordinary_callers_separate(dynamic, unused_object):
    vocab = [
        "<s>",
        "</s>",
        "<call>",
        "<mid>",
        "<skip>",
        "<end>",
        "<bad>",
        "hello",
        "world",
        "x",
        "y",
        "done",
    ]
    source = """leaf ::= [a-z]+
inner ::= TagDispatch(("<end>", leaf), excludes=("<bad>"))
mid_body ::= Token(9, 10) inner
mid ::= TokenTagDispatch((3, mid_body), excludes=(4,))
root ::= TagDispatch(("<call>", mid), excludes=("<bad>"))
"""
    if unused_object:
        object_b = xgr.Grammar.from_json_schema(
            _object("b"), any_order=True, any_whitespace=False, separators=(",", ":")
        )
        source = (
            source.replace("\nroot ::=", "\ndispatch ::=")
            + "root ::= dispatch | unused_json\n"
            + re.sub(r"\broot\b", "unused_json", str(object_b))
        )
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[1])
    compiled = xgr.GrammarCompiler(
        info, max_threads=1, enable_dynamic_compilation=dynamic
    ).compile_grammar(source)
    # The short public mixed-dispatch path must remain in inner's string
    # exclusions; completing its nullable body must not restore mid's token loop.
    _walk(compiled, info, [2, 3, 9], [6], False)
    _walk(compiled, info, [2], [6], True)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("wrappers", [1, 2])
def test_atomic_object_rebase_preserves_zero_context_capture_callers(dynamic, wrappers):
    info = _info(b"xx", ['{"b":1}'])
    object_b = xgr.Grammar.from_json_schema(
        _object("b"), any_order=True, any_whitespace=False, separators=(",", ":")
    )
    source = 'root ::= Token(0) outer | "xxB"\n'
    source += 'wrapped[capture="body"] ::= unused_json "z"\n'
    if wrappers == 1:
        source += "outer ::= wrapped\n"
    else:
        source += 'outer[capture="outer"] ::= wrapped "q"\n'
    source += re.sub(r"\broot\b", "unused_json", str(object_b))
    compiled = xgr.GrammarCompiler(
        info, max_threads=1, enable_dynamic_compilation=dynamic
    ).compile_grammar(source)
    # The wrapper is outside the object scope, but its saved caller is created
    # in the same atomic closure. Preserve this C1-supported continuation to EOS.
    suffix = [257, ord("z") + 1] + ([ord("q") + 1] if wrappers == 2 else [])
    _walk(compiled, info, [0], suffix, True)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("macro", [False, True], ids=["bytes", "macro"])
def test_atomic_prefix_can_continue_into_a_separate_object(dynamic, macro):
    # Token 0 is accepted both atomically and as a partial object A. Object B
    # starts only after the atomic token; its caller must survive the merge.
    prefix, suffix = b'{"a":0', '{"b":1}'
    info = _info(prefix, [suffix])

    def schema_format(schema):
        return {"type": "json_schema", "style": "json", "json_schema": schema, "any_order": True}

    tag = {
        "type": "structural_tag",
        "format": {
            "type": "or",
            "elements": [
                schema_format(_object("a")),
                {
                    "type": "sequence",
                    "elements": [{"type": "token", "token": 0}, schema_format(_object("b"))],
                },
            ],
        },
    }
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_structural_tag(
        json.dumps(tag)
    )
    suffix_ids = [257] if macro else [value + 1 for value in suffix.encode()]
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        for action in ("plain", "fork", "rollback"):
            _walk(grammar, info, [0], suffix_ids, True, action)
        _walk(grammar, info, [0], [ord("}") + 1], True)
        _walk(grammar, info, [value + 1 for value in prefix], suffix_ids, False)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("macro", [False, True], ids=["bytes", "macro"])
@pytest.mark.parametrize("prefix", [b"<", b"<{"])
def test_atomic_and_byte_paths_share_an_object_rule_without_sharing_callers(dynamic, macro, prefix):
    # For a multi-byte prefix, the byte arm enters the same object rule at an
    # intermediate row. The arms have different continuations to expose a return
    # through the wrong caller. Later siblings and array elements reuse that rule.
    # The multi-byte case is an additional row-identity safety control; unlike
    # the structural-tag regression above, it also exposes a baseline limitation.
    tail = '|{"b":2}|[{"b":3},{"b":4}]'
    good, other = '{"b":1}b' + tail, '{"b":1}a' + tail
    info = _info(prefix, [good, other])
    object_b = xgr.Grammar.from_json_schema(
        _object("b"), any_order=True, any_whitespace=False, separators=(",", ":")
    )
    source = (
        'start: ("<" item "a" | <[0]> item "b") "|" item "|" "[" item ("," item)* "]"\n'
        "item: @object_b\n"
    )
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_lark(
        source, named_grammars={"object_b": object_b}
    )
    good_ids = [257] if macro else [value + 1 for value in good.encode()]
    other_ids = [258] if macro else [value + 1 for value in other.encode()]
    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        for action in ("plain", "fork", "rollback"):
            _walk(grammar, info, [0], good_ids, True, action)
        _walk(grammar, info, [0], other_ids, prefix == b"<")


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "budget,pattern,body_ids,body",
    [
        ("max_tokens=1", "/[a-z]*/", [1], b"a"),
        ("max_tokens=1", "/[a-z]+ /", [1, 5], b"a "),
        ("max_chars=0", "/[中]*/", [], b""),
        ("max_chars=1", "/[中]*/", [6], "中".encode()),
    ],
    ids=["token-expiry", "token-relaxation", "char-zero", "char-one-unicode"],
)
def test_budgeted_byte_region_can_force_close_into_an_atomic_tail(
    dynamic, budget, pattern, body_ids, body
):
    info = xgr.TokenizerInfo(
        ['{"中":0}'.encode(), b"a", b"END", b"!", b"<eos>", b" ", "中".encode()], stop_token_ids=[4]
    )
    object_b = xgr.Grammar.from_json_schema(
        _object("中"), any_order=True, any_whitespace=False, separators=(",", ":")
    )
    # Object metadata is present, but this region has only byte edges. Its
    # exhausted budget may close the missing suffix and expose an atomic tail.
    capture = ', capture="body"' if body else ""
    source = (
        "start: @object_b limited <[2]>\n" f'limited[{budget}{capture}, suffix="!"]: {pattern}\n'
    )
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_lark(
        source, named_grammars={"object_b": object_b}
    )
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)

    def allowed(matcher):
        matcher.fill_next_token_bitmask(mask)
        return {
            token
            for token in range(info.vocab_size)
            if int(mask[0, token // 32]) & (1 << (token % 32))
        }

    def feed_body(matcher):
        for token in body_ids:
            available = allowed(matcher)
            assert token in available and 2 not in available
            assert matcher.accept_token(token)

    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        matcher = xgr.GrammarMatcher(grammar)
        assert 0 in allowed(matcher)
        assert matcher.accept_token(0)
        feed_body(matcher)
        for current in (matcher, matcher.fork()):
            before_tail = current.get_captures()
            for _ in range(2):
                assert allowed(current) == {2}
                assert current.get_captures() == before_tail
            assert current.accept_token(2)
            assert current.get_captures() == ([("body", body)] if body else [])
            assert allowed(current) == {4}
            assert current.accept_token(4) and current.is_terminated()
            current.rollback(len(body_ids) + 2)
            assert current.get_captures() == []
            feed_body(current)
            assert allowed(current) == {2}
            assert current.accept_token(2)
            assert current.get_captures() == ([("body", body)] if body else [])


@pytest.mark.parametrize("dynamic", [False, True])
def test_zero_token_budget_between_an_object_and_atomic_tail(dynamic):
    info = xgr.TokenizerInfo(['{"中":0}'.encode(), b"a", b"END", b"<eos>"], stop_token_ids=[3])
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "json_schema",
                    "style": "json",
                    "json_schema": _object("中"),
                    "any_order": True,
                },
                {"type": "any_tokens", "max_tokens": 0},
                {"type": "token", "token": 2},
            ],
        },
    }
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_structural_tag(
        tag
    )
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)

    def allowed(matcher):
        matcher.fill_next_token_bitmask(mask)
        return {
            token
            for token in range(info.vocab_size)
            if int(mask[0, token // 32]) & (1 << (token % 32))
        }

    for grammar in (
        compiled,
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
    ):
        matcher = xgr.GrammarMatcher(grammar)
        assert 0 in allowed(matcher) and matcher.accept_token(0)
        for current in (matcher, matcher.fork()):
            for _ in range(2):
                # Budget rejection follows Fill's enforcement policy on the same matcher.
                assert allowed(current) == {2}
                assert not current.accept_token(1)
                assert allowed(current) == {2} and current.accept_token(2)
                assert allowed(current) == {3}
                assert current.accept_token(3) and current.is_terminated()
                current.rollback(2)


def _scope_first_rules(first):
    choices = {"ab": "first_a | first_b", "ba": "first_b | first_a", "a": "first_a", "b": "first_b"}
    order = ("b", "a") if first == "ba" else ("a", "b")
    return (
        "first ::= "
        + choices[first]
        + "\n"
        + "\n".join(f'first_{key}[object_key="{key}"] ::= "x"' for key in order)
        + "\n"
    )


def _scope_accept(matcher, info, token):
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    matcher.fill_next_token_bitmask(mask)
    allowed = bool(int(mask[0, token // 32]) & (1 << (token % 32)))
    accepted = matcher.accept_token(token)
    assert allowed == accepted, matcher._debug_print_internal_state()
    return accepted


def _scope_check(matcher, info, ids, expected, captures=None):
    for token in ids:
        if not _scope_accept(matcher, info, token):
            assert not expected
            return
    assert expected and matcher.is_terminated()
    if captures is not None:
        assert matcher.get_captures() == captures


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("budget", ["max_chars", "max_tokens"])
def test_forced_completion_preserves_each_object_key_context(dynamic, budget):
    info = _info(b"x", ["xzx"])
    eos, z = info.stop_token_ids[0], ord("z") + 1
    for first in ("ab", "ba", "b", "a"):
        choices = {
            "ab": "first_a | first_b",
            "ba": "first_b | first_a",
            "a": "first_a",
            "b": "first_b",
        }
        order = ("b", "a") if first == "ba" else ("a", "b")
        source = (
            "start: first limited last_a\nfirst: "
            + choices[first]
            + "\n"
            + "\n".join(f'first_{key}: "x"' for key in order)
            + '\nlast_a: "x"\n'
            + f'limited[{budget}=1, suffix="!", capture="body"]: "z"\n'
        )
        grammar = xgr.Grammar.from_lark(source)
        stored = json.loads(grammar.serialize_json())
        root_name = stored["rules"][stored["root_rule_id"]][0]
        source = str(grammar).replace(root_name + " ::=", root_name + "[object_keys] ::=")
        for name, key in (("first_a", "a"), ("first_b", "b"), ("last_a", "a")):
            source = source.replace(name + " ::=", name + f'[object_key="{key}"] ::=')
        compiled = xgr.GrammarCompiler(
            info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=dynamic
        ).compile_grammar(xgr.Grammar.from_ebnf(source, root_rule_name=root_name))
        ids = [0, z, 0, eos]
        _scope_check(xgr.GrammarMatcher(compiled), info, ids, first != "a")
        if first == "ab" and budget == "max_chars" and not dynamic:
            restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
            matcher = xgr.GrammarMatcher(restored)
            assert _scope_accept(matcher, info, 0)
            for current in (matcher, matcher.fork()):
                _scope_check(current, info, ids[1:], True, [("body", b"z")])
                current.rollback(len(ids))
                assert current.get_captures() == []
                _scope_check(current, info, ids, True, [("body", b"z")])
            # A whole token must agree with replay from the complete ambiguous row.
            _walk(restored, info, [], [257], True)


@pytest.mark.parametrize("dynamic", [False, True])
def test_lazy_key_completion_commits_only_its_entry_scope(dynamic):
    info = _info(b"x", ["az", "xaz!"])
    eos, a, z, end = info.stop_token_ids[0], ord("a") + 1, ord("z") + 1, ord("!") + 1
    for first in ("ab", "ba", "a", "b"):
        source = (
            'root[object_keys] ::= first limited "!"\n'
            + _scope_first_rules(first)
            + 'limited[lazy, object_key_raw, capture="key"] ::= "a" | "az"\n'
        )
        compiled = xgr.GrammarCompiler(
            info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=dynamic
        ).compile_grammar(source)
        for body, expected in (([a, z], first != "b"), ([a], first != "a")):
            ids = [0] + body + [end, eos]
            captures = [("key", b"az" if len(body) == 2 else b"a")]
            _scope_check(xgr.GrammarMatcher(compiled), info, ids, expected, captures)
            pure = xgr.GrammarMatcher(compiled)
            assert (
                all(pure.accept_token(token) for token in ids) and pure.is_terminated()
            ) == expected
        if first in ("ab", "b"):
            # Both cache modes and serialized masks must defer a short guarded commit.
            for grammar in (
                compiled,
                xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
            ):
                _scope_check(
                    xgr.GrammarMatcher(grammar),
                    info,
                    [0, 257, end, eos],
                    first != "b",
                    [("key", b"az")],
                )
                _scope_check(
                    xgr.GrammarMatcher(grammar), info, [258, eos], first != "b", [("key", b"az")]
                )
                pure = xgr.GrammarMatcher(grammar)
                assert (pure.accept_token(258) and pure.accept_token(eos)) == (first != "b")
        if first == "ab" and not dynamic:
            restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
            matcher = xgr.GrammarMatcher(restored)
            assert _scope_accept(matcher, info, 0)
            for current in (matcher, matcher.fork()):
                _scope_check(current, info, [257, end, eos], True, [("key", b"az")])
                current.rollback(4)
                _scope_check(current, info, [0, a, z, end, eos], True, [("key", b"az")])


@pytest.mark.parametrize("dynamic", [False, True])
def test_parent_key_rejection_does_not_undo_a_child_lazy_commit(dynamic):
    info = _info(b"x", [])
    eos, a, z, end = info.stop_token_ids[0], ord("a") + 1, ord("z") + 1, ord("!") + 1
    for first in ("a", "b", "ab"):
        source = (
            'root[object_keys] ::= first key "!"\n'
            + _scope_first_rules(first)
            + 'key[object_key_raw] ::= limited\nlimited[lazy] ::= "a" | "az"\n'
        )
        compiled = xgr.GrammarCompiler(
            info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=dynamic
        ).compile_grammar(source)
        _scope_check(xgr.GrammarMatcher(compiled), info, [0, a, z, end, eos], False)
        _scope_check(xgr.GrammarMatcher(compiled), info, [0, a, end, eos], first != "a")


@pytest.mark.parametrize("dynamic", [False, True])
def test_nullable_lazy_key_preserves_literal_rejection_and_no_owner_commit(dynamic):
    info = _info(b"x", ["az", "ax"])
    eos, end = info.stop_token_ids[0], ord("!") + 1
    for first in ("ab", "a", "b", "none"):
        root = "root" if first == "none" else "root[object_keys]"
        rules = _scope_first_rules("b" if first == "none" else first)
        rules = rules.replace('object_key="a"', 'object_key=""')
        source = (
            root
            + ' ::= first limited "!"\n'
            + rules
            + 'limited[lazy, object_key_raw] ::= "" | "az"\n'
        )
        compiled = xgr.GrammarCompiler(
            info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=dynamic
        ).compile_grammar(source)
        for grammar in (
            compiled,
            xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info),
        ):
            _scope_check(
                xgr.GrammarMatcher(grammar), info, [0, 257, end, eos], first in ("ab", "a")
            )
            _scope_check(xgr.GrammarMatcher(grammar), info, [0, 258, end, eos], False)
            _scope_check(xgr.GrammarMatcher(grammar), info, [0, end, eos], first != "a")
