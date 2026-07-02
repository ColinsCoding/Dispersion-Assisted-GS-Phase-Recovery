"""Tests for dgs/grammar_pts.py -- Photonic Description Language parser"""
import math
import pytest

try:
    from dgs.grammar_pts import (
        tokenize, parse, evaluate, grammar_summary, Token, PDL_BNF,
        SystemNode, ComponentNode, PDLParser,
    )
except ImportError:
    from grammar_pts import (
        tokenize, parse, evaluate, grammar_summary, Token, PDL_BNF,
        SystemNode, ComponentNode, PDLParser,
    )


# ── Tokenizer tests ─────────────────────────────────────────────────────

class TestTokenizer:
    def test_single_component(self):
        toks = tokenize("FIBER")
        kinds = [t.kind for t in toks]
        assert Token.COMPONENT in kinds

    def test_arrow_token(self):
        toks = tokenize("FIBER -> EOM")
        kinds = [t.kind for t in toks]
        assert Token.ARROW in kinds

    def test_param_tokens(self):
        toks = tokenize("FIBER(D=17,L=5)")
        kinds = [t.kind for t in toks]
        assert Token.LPAREN in kinds
        assert Token.RPAREN in kinds
        assert Token.EQUALS in kinds
        assert Token.COMMA in kinds
        assert Token.NUMBER in kinds

    def test_ident_token_for_param_name(self):
        toks = tokenize("FIBER(D=17)")
        ident_toks = [t for t in toks if t.kind == Token.IDENT]
        assert any(t.value == 'D' for t in ident_toks)

    def test_number_float(self):
        toks = tokenize("ADC(fs=2.5)")
        num_toks = [t for t in toks if t.kind == Token.NUMBER]
        assert any(abs(float(t.value) - 2.5) < 1e-9 for t in num_toks)

    def test_scientific_notation(self):
        toks = tokenize("LASER(P=1e-3)")
        num_toks = [t for t in toks if t.kind == Token.NUMBER]
        assert any(abs(float(t.value) - 1e-3) < 1e-12 for t in num_toks)

    def test_eof_present(self):
        toks = tokenize("FIBER")
        assert toks[-1].kind == Token.EOF

    def test_whitespace_skipped(self):
        toks1 = tokenize("FIBER(D=17)")
        toks2 = tokenize("FIBER( D = 17 )")
        kinds1 = [t.kind for t in toks1]
        kinds2 = [t.kind for t in toks2]
        assert kinds1 == kinds2

    def test_unknown_char_raises(self):
        with pytest.raises(SyntaxError):
            tokenize("FIBER @ EOM")

    def test_all_component_names(self):
        for name in ['LASER','EDFA','FIBER','EOM','PD','ADC','GS','GRATING']:
            toks = tokenize(name)
            assert toks[0].kind == Token.COMPONENT
            assert toks[0].value == name

    def test_position_tracked(self):
        toks = tokenize("FIBER")
        assert toks[0].pos == 0


# ── Parser tests ─────────────────────────────────────────────────────────

class TestParser:
    def test_single_component_parses(self):
        ast = parse("FIBER")
        assert isinstance(ast, SystemNode)
        assert len(ast.components) == 1
        assert ast.components[0].name == 'FIBER'

    def test_two_components(self):
        ast = parse("FIBER -> EOM")
        assert len(ast.components) == 2

    def test_three_components(self):
        ast = parse("EDFA -> FIBER -> PD")
        assert len(ast.components) == 3

    def test_params_parsed(self):
        ast = parse("FIBER(D=17,L=5)")
        c = ast.components[0]
        assert c.params['D'] == pytest.approx(17.0)
        assert c.params['L'] == pytest.approx(5.0)

    def test_no_params(self):
        ast = parse("EOM")
        assert ast.components[0].params == {}

    def test_full_coppinger_system(self):
        s = ("LASER(P=0.001) -> EDFA(G=30) -> FIBER(D=17,L=5) -> "
             "EOM(Vpi=3.5) -> FIBER(D=17,L=45) -> PD(R=0.8) -> ADC(fs=2)")
        ast = parse(s)
        assert len(ast.components) == 7

    def test_gs_component_parsed(self):
        ast = parse("GS(n=50,D=5000)")
        c = ast.components[0]
        assert c.params['n'] == 50
        assert c.params['D'] == 5000

    def test_missing_paren_raises(self):
        with pytest.raises(SyntaxError):
            parse("FIBER(D=17")

    def test_extra_token_raises(self):
        with pytest.raises(SyntaxError):
            parse("FIBER EDFA")   # no arrow between them

    def test_scientific_param_parsed(self):
        ast = parse("LASER(P=1e-3)")
        assert ast.components[0].params['P'] == pytest.approx(1e-3)

    def test_component_node_repr(self):
        ast = parse("FIBER(D=17,L=5)")
        r = repr(ast.components[0])
        assert 'FIBER' in r and 'D' in r


# ── Evaluator tests ──────────────────────────────────────────────────────

class TestEvaluator:
    def test_coppinger_stretch_M10(self):
        s = ("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45)")
        r = evaluate(s)
        assert r['stretch']['M'] == pytest.approx(10.0, rel=0.01)

    def test_B_RF_coppinger(self):
        s = "FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45) -> ADC(fs=2)"
        r = evaluate(s)
        # M=10, fs=2 -> B_RF = M*fs/2 = 10*2/2 = 10 GHz
        assert r['stretch']['B_RF_GHz'] == pytest.approx(10.0, rel=0.01)

    def test_DL_pre_correct(self):
        s = "FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45)"
        r = evaluate(s)
        assert r['fiber']['DL_pre_ps_nm'] == pytest.approx(17*5, rel=0.01)

    def test_DL_post_correct(self):
        s = "FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45)"
        r = evaluate(s)
        assert r['fiber']['DL_post_ps_nm'] == pytest.approx(17*45, rel=0.01)

    def test_M_equals_1_no_fiber(self):
        r = evaluate("EOM -> ADC")
        assert r['stretch']['M'] == pytest.approx(1.0, rel=0.01)
        assert any('M <= 1' in w for w in r['warnings'])

    def test_eom_position_tracked(self):
        r = evaluate("EDFA -> FIBER(D=17,L=5) -> EOM -> PD")
        assert r['fiber']['eom_position'] == 2

    def test_no_eom_warning(self):
        r = evaluate("FIBER -> PD -> ADC")
        assert any('EOM' in w for w in r['warnings'])

    def test_gs_D_too_small_warns(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45) -> GS(n=50,D=100)")
        assert any('|D|' in w or 'D=100' in w for w in r['warnings'])

    def test_gs_n_too_small_warns(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45) -> GS(n=10,D=5000)")
        assert any('n=10' in w or 'n_iter' in w or 'n>=50' in w for w in r['warnings'])

    def test_H_total_all_pass(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45)")
        assert r['H_total']['all_pass'] is True

    def test_H_total_phase_nonzero(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45)")
        phases = r['H_total']['H_phase_rad']
        assert max(abs(p) for p in phases) > 0

    def test_valid_system(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45) -> ADC(fs=2)")
        assert r['valid'] is True

    def test_component_count(self):
        r = evaluate("LASER -> EDFA -> FIBER -> EOM -> FIBER -> PD -> ADC -> GS")
        assert r['n_components'] == 8

    def test_SNR_key_present(self):
        r = evaluate("LASER(P=0.001) -> FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45) -> PD(R=0.8) -> ADC")
        assert 'SNR_dB' in r['power']

    def test_DCF_negative_D(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> DCF(D=-85,L=1)")
        # DCF has negative D: post-EOM DL contribution is negative
        assert r['fiber']['DL_post_ps_nm'] == pytest.approx(-85*1, rel=0.01)

    def test_SPOOL_component_works(self):
        r = evaluate("SPOOL(D=17,L=5) -> EOM -> SPOOL(D=17,L=25)")
        assert r['stretch']['M'] == pytest.approx(6.0, rel=0.05)

    def test_minimal_system_M2(self):
        r = evaluate("FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=5)")
        assert r['stretch']['M'] == pytest.approx(2.0, rel=0.01)


# ── Grammar documentation tests ──────────────────────────────────────────

class TestGrammarDoc:
    def test_BNF_has_system_rule(self):
        assert '<system>' in PDL_BNF

    def test_BNF_has_component_rule(self):
        assert '<component>' in PDL_BNF

    def test_grammar_summary_returns_dict(self):
        gs = grammar_summary()
        assert isinstance(gs, dict)

    def test_grammar_class_CFL(self):
        gs = grammar_summary()
        assert 'Context-free' in gs['language_class']

    def test_parser_class_LL1(self):
        gs = grammar_summary()
        assert 'LL(1)' in gs['parser_class']

    def test_examples_nonempty(self):
        gs = grammar_summary()
        assert len(gs['examples']) > 0

    def test_each_example_parses(self):
        gs = grammar_summary()
        for ex in gs['examples']:
            r = evaluate(ex)
            assert r['valid'] is True

    def test_token_types_listed(self):
        gs = grammar_summary()
        assert len(gs['token_types']) >= 7
