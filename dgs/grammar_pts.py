"""
Photonic System Description Language (PDL) -- Formal Grammar + Parser
======================================================================
Implements a context-free grammar (BNF/EBNF) for describing and
evaluating photonic time-stretch systems.

BNF GRAMMAR (PDL v1.0):
  <system>    ::= <component> ('->' <component>)*
  <component> ::= <name> <params>?
  <name>      ::= 'EDFA' | 'FIBER' | 'EOM' | 'PD' | 'ADC' | 'GS'
                | 'LASER' | 'GRATING' | 'COUPLER' | 'MZI' | 'FILTER'
  <params>    ::= '(' <param> (',' <param>)* ')'
  <param>     ::= <identifier> '=' <number>
  <identifier>::= [A-Za-z_][A-Za-z0-9_]*
  <number>    ::= [0-9]+ ('.' [0-9]+)? (('e'|'E') ('+''|'-')? [0-9]+)?

EXAMPLE SYSTEMS (Coppinger/Jalali 1999):
  "LASER(P=0.1) -> EDFA(G=30) -> FIBER(D=17,L=5) ->
   EOM(Vpi=3.5) -> FIBER(D=17,L=45) -> PD(R=0.8) -> ADC(fs=2) -> GS(n=50)"

The parser computes:
  - Stretch factor M from FIBER components
  - SNR budget through the chain
  - H(f) = exp(j*pi*D_total*f^2) transfer function
  - System validation (physical constraints)

FORMAL LANGUAGE THEORY:
  PDL is a context-free language (CFL).
  Recognized by a pushdown automaton (PDA).
  The grammar is LL(1) -- parse with 1 lookahead token.

  Parse tree example for "FIBER(D=17,L=5)":
    <component>
      <name>: 'FIBER'
      <params>:
        <param>: D=17
        <param>: L=5

  This IS the chain rule:
    evaluate(system) = compose(evaluate(c_n), ..., evaluate(c_1))
    where compose = matrix multiplication of transfer functions
"""
import re
import math
import numpy as np


# ============================================================
# Lexer (Tokenizer)
# ============================================================

class Token:
    """A single lexical token."""
    COMPONENT = 'COMPONENT'
    LPAREN    = 'LPAREN'
    RPAREN    = 'RPAREN'
    ARROW     = 'ARROW'
    EQUALS    = 'EQUALS'
    COMMA     = 'COMMA'
    NUMBER    = 'NUMBER'
    IDENT     = 'IDENT'
    EOF       = 'EOF'

    def __init__(self, kind, value, pos):
        self.kind = kind; self.value = value; self.pos = pos

    def __repr__(self):
        return f'Token({self.kind}, {self.value!r}, pos={self.pos})'


COMPONENT_NAMES = {
    'LASER', 'EDFA', 'FIBER', 'EOM', 'PD', 'ADC', 'GS',
    'GRATING', 'COUPLER', 'MZI', 'FILTER', 'ISOLATOR', 'WDM',
    'CIRCULATOR', 'SPOOL', 'DCF', 'SMF', 'SOA', 'VNA',
}

_TOKEN_PATTERNS = [
    (r'->',              Token.ARROW),
    (r'\(',              Token.LPAREN),
    (r'\)',              Token.RPAREN),
    (r',',               Token.COMMA),
    (r'=',               Token.EQUALS),
    (r'-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?', Token.NUMBER),
    (r'[A-Za-z_][A-Za-z0-9_]*', Token.IDENT),
    (r'[ \t\n\r]+',     None),   # whitespace: skip
]

_MASTER_RE = re.compile('|'.join(f'({p})' for p, _ in _TOKEN_PATTERNS))


def tokenize(text):
    """Convert PDL string into token list."""
    tokens = []
    pos = 0
    while pos < len(text):
        m = _MASTER_RE.match(text, pos)
        if not m:
            raise SyntaxError(
                f"PDL Lexer: unexpected character {text[pos]!r} at position {pos}"
            )
        for i, (pattern, kind) in enumerate(_TOKEN_PATTERNS):
            if m.group(i+1) is not None:
                val = m.group(i+1)
                if kind is not None:
                    # Classify IDENT as COMPONENT if it's a known component name
                    if kind == Token.IDENT and val.upper() in COMPONENT_NAMES:
                        kind = Token.COMPONENT
                        val = val.upper()
                    tokens.append(Token(kind, val, pos))
                break
        pos = m.end()
    tokens.append(Token(Token.EOF, None, pos))
    return tokens


# ============================================================
# AST Nodes
# ============================================================

class SystemNode:
    """Root node: list of components connected by ->"""
    def __init__(self, components):
        self.components = components   # list of ComponentNode

    def __repr__(self):
        return f"System([{', '.join(str(c) for c in self.components)}])"


class ComponentNode:
    """A single component: name + dict of parameters."""
    def __init__(self, name, params):
        self.name = name          # str
        self.params = params      # dict {str: float}

    def __repr__(self):
        p = ', '.join(f'{k}={v}' for k,v in self.params.items())
        return f"{self.name}({p})" if p else self.name


# ============================================================
# Recursive-Descent Parser (LL(1))
# ============================================================

class PDLParser:
    """
    Recursive descent parser for Photonic Description Language.

    Grammar:
      system     -> component (ARROW component)*
      component  -> COMPONENT params?
      params     -> LPAREN param (COMMA param)* RPAREN
      param      -> IDENT EQUALS NUMBER

    Each method corresponds to one grammar production rule.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def consume(self, expected_kind=None):
        tok = self.tokens[self.pos]
        if expected_kind and tok.kind != expected_kind:
            raise SyntaxError(
                f"PDL Parser: expected {expected_kind} but got {tok.kind} "
                f"({tok.value!r}) at position {tok.pos}"
            )
        self.pos += 1
        return tok

    def parse_system(self):
        """system -> component (ARROW component)*"""
        components = [self.parse_component()]
        while self.peek().kind == Token.ARROW:
            self.consume(Token.ARROW)
            components.append(self.parse_component())
        if self.peek().kind != Token.EOF:
            tok = self.peek()
            raise SyntaxError(
                f"PDL Parser: unexpected token {tok.kind}({tok.value!r}) at {tok.pos}"
            )
        return SystemNode(components)

    def parse_component(self):
        """component -> COMPONENT params?"""
        name_tok = self.consume(Token.COMPONENT)
        name = name_tok.value
        params = {}
        if self.peek().kind == Token.LPAREN:
            params = self.parse_params()
        return ComponentNode(name, params)

    def parse_params(self):
        """params -> LPAREN param (COMMA param)* RPAREN"""
        self.consume(Token.LPAREN)
        params = {}
        params.update(self.parse_param())
        while self.peek().kind == Token.COMMA:
            self.consume(Token.COMMA)
            params.update(self.parse_param())
        self.consume(Token.RPAREN)
        return params

    def parse_param(self):
        """param -> IDENT EQUALS NUMBER"""
        key_tok = self.consume(Token.IDENT)
        self.consume(Token.EQUALS)
        val_tok = self.consume(Token.NUMBER)
        return {key_tok.value: float(val_tok.value)}


def parse(system_string):
    """Parse a PDL system string -> SystemNode AST."""
    tokens = tokenize(system_string)
    parser = PDLParser(tokens)
    return parser.parse_system()


# ============================================================
# Evaluator: compute system parameters from AST
# ============================================================

c_light = 2.998e8

class SystemEvaluator:
    """
    Evaluate a parsed photonic system AST.

    Computes:
      - Stretch factor M = 1 + sum(D2*L2) / sum(D1*L1)
        (where D1*L1 = pre-EOM fiber GVD*length,
               D2*L2 = post-EOM fiber GVD*length)
      - Captured RF bandwidth = M * f_ADC / 2
      - Total dispersion D_eff = sum(D_i * L_i)
      - SNR chain (optical power budget)
      - H(f) = exp(j*pi*D_eff*f^2)
      - Validation warnings (physical constraints)

    CHAIN RULE CONNECTION:
      H_total(f) = H_N(f) * H_{N-1}(f) * ... * H_1(f)
      For dispersion: H_i(f) = exp(j*phi_i(f))
      Total: H_total(f) = exp(j*sum(phi_i(f))) = exp(j*pi*D_eff*f^2)
      This IS the chain rule for transfer functions: compose by multiplication.

    FORMAL LANGUAGE PROPERTIES:
      PDL is regular in component names (finite alphabet of keywords).
      PDL has context-free structure (nested params with matching parens).
      The evaluator is a tree-walking interpreter (structural recursion).
      Typing rules (e.g., FIBER must have D and L) are context-sensitive.
    """

    # Default component parameters
    DEFAULTS = {
        'LASER':      {'P': 0.001, 'lambda': 1550.0},      # 1 mW, 1550 nm
        'EDFA':       {'G': 20.0, 'NF': 5.0, 'P_sat': 100.0},  # 20 dB gain
        'FIBER':      {'D': 17.0, 'L': 1.0, 'alpha': 0.2},  # SMF-28 defaults
        'SMF':        {'D': 17.0, 'L': 1.0, 'alpha': 0.2},
        'DCF':        {'D': -85.0,'L': 1.0, 'alpha': 0.5}, # dispersion compensating
        'EOM':        {'Vpi': 3.5, 'IL': 3.0, 'BW': 40.0}, # insertion loss dB
        'MZI':        {'Vpi': 3.5, 'IL': 3.0},
        'PD':         {'R': 0.8,  'BW': 50.0, 'NEP': 1e-12},
        'ADC':        {'fs': 1.0, 'ENOB': 8.0, 'bits': 8},
        'GS':         {'n': 50,   'D': 5000.0},
        'GRATING':    {'lpmm': 600.0, 'eta': 0.9},
        'COUPLER':    {'split': 0.5},
        'FILTER':     {'BW': 10.0, 'loss': 1.0},
        'ISOLATOR':   {'iso': 40.0, 'IL': 0.5},
        'WDM':        {'loss': 1.0},
        'CIRCULATOR': {'iso': 40.0, 'IL': 1.0},
        'SOA':        {'G': 15.0, 'NF': 8.0},
        'VNA':        {'f_start': 0.1, 'f_stop': 50.0},
        'SPOOL':      {'D': 17.0, 'L': 10.0, 'alpha': 0.2},
    }

    def __init__(self, ast):
        self.ast = ast
        self.warnings = []
        self.errors   = []

    def _get(self, comp, key):
        """Get parameter with default fallback."""
        return comp.params.get(key, self.DEFAULTS.get(comp.name, {}).get(key, 0.0))

    def evaluate(self):
        components = self.ast.components
        n = len(components)

        # ── Locate key components ──────────────────────────
        eom_idx = None
        for i, c in enumerate(components):
            if c.name in ('EOM', 'MZI'):
                eom_idx = i; break

        # ── Fiber dispersion accounting ─────────────────────
        # Pre-EOM fibers = L1 elements (chirp the pulse)
        # Post-EOM fibers = L2 elements (time-stretch the signal)
        fiber_names = {'FIBER','SMF','DCF','SPOOL'}
        DL_pre = 0.0; DL_post = 0.0   # ps/nm (D*L products)
        total_loss_dB = 0.0

        for i, c in enumerate(components):
            if c.name in fiber_names:
                D = self._get(c, 'D'); L = self._get(c, 'L')
                alpha = self._get(c, 'alpha')
                DL = D * L   # ps/nm * km = ps*km/nm... no: D [ps/(nm*km)] * L[km] = ps/nm. Correct.
                loss = alpha * L   # dB
                if eom_idx is None or i < eom_idx:
                    DL_pre += DL
                else:
                    DL_post += DL
                total_loss_dB += loss

            elif c.name == 'EDFA':
                G = self._get(c, 'G'); NF = self._get(c, 'NF')
                total_loss_dB -= G   # gain reduces effective loss

            elif c.name in ('EOM', 'MZI'):
                total_loss_dB += self._get(c, 'IL')

            elif c.name == 'COUPLER':
                split = self._get(c, 'split')
                total_loss_dB += -10*math.log10(max(split, 1e-10))

        # ── Stretch factor ──────────────────────────────────
        if DL_pre > 0:
            M = (DL_pre + DL_post) / DL_pre
        else:
            M = 1.0
            self.warnings.append("No pre-EOM fiber found; M = 1 (no stretch)")

        # ── ADC parameters ──────────────────────────────────
        adc = next((c for c in components if c.name == 'ADC'), None)
        fs_GHz = self._get(adc, 'fs') if adc else 1.0
        ENOB   = self._get(adc, 'ENOB') if adc else 8.0
        B_RF_GHz = M * fs_GHz / 2

        # ── Optical power budget ────────────────────────────
        laser = next((c for c in components if c.name == 'LASER'), None)
        P_laser_dBm = 10*math.log10(self._get(laser,'P')*1e3) if laser else 0.0
        P_pd_dBm = P_laser_dBm - total_loss_dB

        # ── SNR estimate (simplified: RIN-limited) ──────────
        pd = next((c for c in components if c.name == 'PD'), None)
        R_resp = self._get(pd, 'R') if pd else 0.8
        P_pd_W = 10**((P_pd_dBm)/10)*1e-3
        I_ph = R_resp * max(P_pd_W, 1e-20)
        RIN_dBHz = -140.0
        B_noise_Hz = B_RF_GHz*1e9 / max(2*abs(M), 1e-10)
        q_e = 1.602e-19; kB = 1.381e-23
        RIN_lin = 10**(RIN_dBHz/10)
        R_load = 50.0; T_K = 300.0
        noise_sq = (2*q_e*I_ph*B_noise_Hz +
                    RIN_lin*B_noise_Hz*I_ph**2 +
                    4*kB*T_K*B_noise_Hz/R_load)
        SNR_dB = 10*math.log10(max(I_ph**2/max(noise_sq, 1e-60), 1e-30))
        ENOB_sys = (SNR_dB - 1.76)/6.02

        # ── H(f) transfer function ──────────────────────────
        f_arr = np.linspace(-B_RF_GHz/2, B_RF_GHz/2, 1000) * 1e9
        D_total = DL_pre + DL_post   # ps/nm
        # Convert: D [ps/nm] (= D*L product) relates to beta2 via:
        # phi(f) = pi * (D_total * 1e-12 / 1e-9) * f^2 / (c/lambda^2)... simplified:
        # Use the dimensionless form: phi(f) ~ pi * (D_total/D0) * (f/f0)^2
        # Actual: H(f) = exp(j*pi*beta2*L*(2*pi*f)^2), D_total in ps/nm:
        # beta2*L = -(lambda^2/(2*pi*c)) * D_total [ps*nm] = -(lambda^2/(2*pi*c)) * D_total * 1e-21
        lambda0_m = 1550e-9
        DL_SI = D_total * 1e-12 * 1e-9   # [s/nm * nm] = ... D [ps/(nm*km)]*L[km] = ps/nm; D_total [ps/nm]
        # D_total is D*L [ps/nm]. In SI: D*L [s/m] = D_total * 1e-12 / 1e-9 = D_total * 1e-3 s/m
        # beta2*L = -(lambda^2/(2*pi*c)) * D_total_SI
        D_total_SI = D_total * 1e-12 / 1e-9   # s/m (the group delay per wavelength, per unit length, times length)
        beta2L = -(lambda0_m**2/(2*np.pi*c_light)) * D_total_SI   # s^2
        phi_arr = np.pi * beta2L * (2*np.pi*f_arr)**2   # rad
        H_total = np.exp(1j*phi_arr)

        # ── Validation ──────────────────────────────────────
        gs = next((c for c in components if c.name == 'GS'), None)
        if gs:
            D_gs = self._get(gs, 'D')
            if abs(D_gs) < 5000:
                self.warnings.append(
                    f"GS(D={D_gs}) < 5000 -- GS may not converge (need |D|>=5000)"
                )
            n_iter = self._get(gs, 'n')
            if n_iter < 50:
                self.warnings.append(
                    f"GS(n={n_iter}) < 50 -- recommend n>=50 iterations"
                )
        if M <= 1.0:
            self.warnings.append("M <= 1: no photonic time-stretch advantage")
        if eom_idx is None:
            self.warnings.append("No EOM found: signal cannot be imprinted on optical carrier")

        return {
            'system_string': str(self.ast),
            'components': [str(c) for c in components],
            'n_components': n,
            'fiber': {
                'DL_pre_ps_nm': float(DL_pre),
                'DL_post_ps_nm': float(DL_post),
                'D_total_ps_nm': float(D_total),
                'eom_position': eom_idx,
            },
            'stretch': {
                'M': float(M),
                'formula': 'M = (DL_pre + DL_post) / DL_pre = 1 + DL_post/DL_pre',
                'B_RF_GHz': float(B_RF_GHz),
            },
            'ADC': {
                'fs_GHz': float(fs_GHz),
                'B_captured_GHz': float(B_RF_GHz),
                'ENOB_nominal': float(ENOB),
                'ENOB_system': float(ENOB_sys),
                'note': f'Captures {B_RF_GHz:.1f} GHz with {fs_GHz:.0f} Gsample/s',
            },
            'power': {
                'P_laser_dBm': float(P_laser_dBm),
                'total_loss_dB': float(total_loss_dB),
                'P_pd_dBm': float(P_pd_dBm),
                'I_photocurrent_mA': float(I_ph*1e3),
                'SNR_dB': float(SNR_dB),
            },
            'H_total': {
                'f_GHz': (f_arr/1e9).tolist(),
                'H_phase_rad': phi_arr.tolist(),
                'H_mag': np.abs(H_total).tolist(),
                'formula': f'H(f) = exp(j*pi*beta2*L*(2*pi*f)^2)  [D_total={D_total:.0f} ps/nm]',
                'all_pass': True,
            },
            'warnings': self.warnings,
            'errors': self.errors,
            'valid': len(self.errors) == 0,
        }


def evaluate(system_string):
    """Parse and evaluate a PDL system string."""
    ast = parse(system_string)
    ev = SystemEvaluator(ast)
    return ev.evaluate()


# ============================================================
# Grammar Documentation (BNF formal spec)
# ============================================================

PDL_BNF = """
PDL v1.0 -- Photonic Description Language (BNF)
================================================

<system>       ::= <component> ('->' <component>)*

<component>    ::= <comp_name> <params>?

<comp_name>    ::= 'LASER' | 'EDFA'  | 'SOA'   | 'FIBER' | 'SMF'
               |   'DCF'   | 'SPOOL' | 'EOM'   | 'MZI'   | 'PD'
               |   'ADC'   | 'GS'    | 'COUPLER'| 'WDM'  | 'FILTER'
               |   'GRATING'| 'ISOLATOR' | 'CIRCULATOR' | 'VNA'

<params>       ::= '(' <param> (',' <param>)* ')'

<param>        ::= <identifier> '=' <number>

<identifier>   ::= [A-Za-z_][A-Za-z0-9_]*

<number>       ::= <integer> | <float> | <scientific>
<integer>      ::= [0-9]+
<float>        ::= [0-9]+ '.' [0-9]+
<scientific>   ::= (<integer> | <float>) ('e'|'E') ('+'|'-')? [0-9]+

WELL-KNOWN PARAMETER NAMES:
  D    -- GVD [ps/(nm*km)]      (FIBER, SMF, DCF, SPOOL)
  L    -- length [km]            (FIBER, SMF, DCF, SPOOL)
  alpha-- attenuation [dB/km]    (FIBER)
  G    -- gain [dB]              (EDFA, SOA)
  NF   -- noise figure [dB]      (EDFA, SOA)
  Vpi  -- half-wave voltage [V]  (EOM, MZI)
  IL   -- insertion loss [dB]    (EOM, MZI, ISOLATOR)
  R    -- responsivity [A/W]     (PD)
  BW   -- bandwidth [GHz]        (PD, EOM, FILTER)
  fs   -- sample rate [Gsample/s](ADC)
  ENOB -- effective # of bits    (ADC)
  n    -- iterations             (GS)
  P    -- power [W]              (LASER)
  lambda -- wavelength [nm]      (LASER)
  split-- coupling ratio [0..1]  (COUPLER)
  lpmm -- lines/mm               (GRATING)

FORMAL LANGUAGE PROPERTIES:
  Type:          Context-free (generated by CFG above)
  Automaton:     Pushdown automaton (PDA) -- stack tracks paren nesting
  Deterministic: Yes (LL(1) parser with 1 lookahead token)
  Ambiguous:     No -- unique parse tree for every valid string
  Complexity:    O(n) parsing -- linear in token count
  Expressiveness: Cannot express feedback loops (would require context-sensitivity)
                  Cannot express wavelength-dependent routing (cross-referencing)
  Extensions:    PDL v2.0 could add: parallel branches '|', loops 'LOOP(n) {...}'

CHAIN RULE IN THE GRAMMAR:
  evaluate(system) = fold_right(compose, [evaluate(c) for c in components])
  where compose(H1, H2) = H1 * H2  (transfer function multiplication)
  This is the function composition law -- the chain rule for linear systems.
"""


def grammar_summary():
    """Return grammar documentation and examples."""
    examples = [
        # Coppinger 1999 demonstration system
        "LASER(P=0.001) -> EDFA(G=30) -> FIBER(D=17,L=5) -> "
        "EOM(Vpi=3.5) -> FIBER(D=17,L=45) -> PD(R=0.8) -> "
        "ADC(fs=2) -> GS(n=50,D=5000)",

        # Minimal time-stretch system
        "EDFA(G=20) -> SMF(D=17,L=1) -> EOM -> SMF(D=17,L=9) -> PD -> ADC(fs=1)",

        # STEAM camera front-end
        "LASER -> EDFA(G=20) -> GRATING(lpmm=600) -> FIBER(D=17,L=5) -> "
        "EOM(BW=40) -> FIBER(D=17,L=45) -> PD(BW=50) -> ADC(fs=1)",

        # RogueGuard system
        "LASER(P=0.01,lambda=1550) -> EDFA(G=20,NF=5) -> "
        "SPOOL(D=17,L=5) -> EOM(Vpi=3.5,IL=3,BW=40) -> "
        "SPOOL(D=17,L=25) -> PD(R=0.8,BW=50) -> ADC(fs=1,ENOB=8) -> "
        "GS(n=50,D=5000)",
    ]
    return {
        'BNF': PDL_BNF,
        'examples': examples,
        'token_types': [
            'COMPONENT (FIBER, EDFA, EOM, ...)',
            'LPAREN (, RPAREN )',
            'ARROW ->',
            'EQUALS =',
            'COMMA ,',
            'NUMBER (float/int/scientific)',
            'IDENT (parameter name: D, L, G, ...)',
            'EOF',
        ],
        'parser_class': 'LL(1) recursive descent',
        'language_class': 'Context-free (CFL)',
        'automaton': 'Pushdown automaton (PDA)',
    }


def demo():
    print("=== PHOTONIC SYSTEM DESCRIPTION LANGUAGE (PDL) ===\n")

    # Print grammar summary
    gs = grammar_summary()
    print("--- Grammar Properties ---")
    print(f"  Class: {gs['language_class']}")
    print(f"  Parser: {gs['parser_class']}")
    print(f"  Automaton: {gs['automaton']}")
    print(f"  Token types: {len(gs['token_types'])}")

    # Tokenize example
    ex1 = "FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45)"
    print(f"\n--- Tokenize: '{ex1}' ---")
    toks = tokenize(ex1)
    for t in toks[:-1]:
        print(f"  {t}")

    # Parse and evaluate: Coppinger 1999
    print("\n--- Coppinger 1999 System ---")
    coppinger = (
        "LASER(P=0.001) -> EDFA(G=30) -> FIBER(D=17,L=5) -> "
        "EOM(Vpi=3.5,IL=3) -> FIBER(D=17,L=45) -> PD(R=0.8) -> "
        "ADC(fs=2,ENOB=8) -> GS(n=50,D=5000)"
    )
    result = evaluate(coppinger)
    print(f"  Components: {result['n_components']}")
    print(f"  DL_pre  = {result['fiber']['DL_pre_ps_nm']:.0f} ps/nm")
    print(f"  DL_post = {result['fiber']['DL_post_ps_nm']:.0f} ps/nm")
    print(f"  M = {result['stretch']['M']:.1f}  (paper: 10)")
    print(f"  B_RF = {result['stretch']['B_RF_GHz']:.0f} GHz  (paper: 10 GHz with f_s=2)")
    print(f"  H(f): {result['H_total']['formula']}")
    print(f"  SNR = {result['power']['SNR_dB']:.1f} dB")
    print(f"  ENOB_system = {result['ADC']['ENOB_system']:.1f} bits")
    if result['warnings']:
        for w in result['warnings']:
            print(f"  WARNING: {w}")

    # Minimal system
    print("\n--- Minimal System (M=2) ---")
    minimal = "EDFA -> FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=5) -> PD -> ADC(fs=1)"
    r2 = evaluate(minimal)
    print(f"  M = {r2['stretch']['M']:.1f}  B_RF = {r2['stretch']['B_RF_GHz']:.1f} GHz")

    # RogueGuard
    print("\n--- RogueGuard System ---")
    rogue = (
        "LASER(P=0.01) -> EDFA(G=20,NF=5) -> SPOOL(D=17,L=5) -> "
        "EOM(Vpi=3.5,IL=3,BW=40) -> SPOOL(D=17,L=25) -> "
        "PD(R=0.8) -> ADC(fs=1,ENOB=8) -> GS(n=50,D=5000)"
    )
    r3 = evaluate(rogue)
    print(f"  M = {r3['stretch']['M']:.1f}  B_RF = {r3['stretch']['B_RF_GHz']:.1f} GHz")
    print(f"  P at PD = {r3['power']['P_pd_dBm']:.1f} dBm")

    print("\n=== PDL GRAMMAR COMPLETE ===")


if __name__ == '__main__':
    demo()
