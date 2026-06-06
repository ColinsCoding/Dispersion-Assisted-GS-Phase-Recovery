"""
_repl_grammar_regex.py

S1: Formal grammar -- Chomsky hierarchy, BNF/EBNF, parse trees
S2: Regex theory -- regular languages = Type-3 grammars
S3: Python regex: first name, phone number, password validation
S4: C header validation.h (printed as string -- POSIX regex.h)
S5: Terminal UI sketch in C (ncurses-style, printed)
"""

import re

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: FORMAL GRAMMAR
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 1: FORMAL GRAMMAR -- CHOMSKY HIERARCHY")
print(SEP)

print("""
  A GRAMMAR G = (V, T, P, S) where:
    V = non-terminal symbols  (variables, written UPPER)
    T = terminal symbols      (actual characters/tokens, written lower)
    P = production rules      (rewriting rules V -> ...)
    S = start symbol

  CHOMSKY HIERARCHY:
  Type  Name                  Recognizer       Example
  ---------------------------------------------------------------
  0     Unrestricted          Turing machine   any computable language
  1     Context-sensitive     Linear-bounded   a^n b^n c^n
  2     Context-free (CFG)    Pushdown automaton  a^n b^n, arithmetic
  3     Regular               Finite automaton    email, phone, identifiers

  KEY INSIGHT:
    Regular expression = finite automaton = Type-3 grammar
    They are ALL the same model of computation.
    Regex cannot count balanced brackets: ((())) is NOT regular.
    For balanced parens you NEED a CFG (pushdown, uses a stack).

  BNF (Backus-Naur Form):
    <expr>   ::= <term> | <expr> "+" <term>
    <term>   ::= <factor> | <term> "*" <factor>
    <factor> ::= "(" <expr> ")" | <number>
    <number> ::= <digit> | <number> <digit>
    <digit>  ::= "0" | "1" | "2" | ... | "9"

  EBNF (Extended BNF, adds *, +, ?, [ ]):
    expr   = term { ("+" | "-") term } ;
    term   = factor { ("*" | "/") factor } ;
    factor = "(" expr ")" | number ;
    number = digit { digit } ;
    digit  = "0" | "1" | ... | "9" ;

  PARSE TREE for  3 + 4 * 2:
    expr
    +-- term (3)
    +-- "+"
    +-- term
        +-- factor (4)
        +-- "*"
        +-- factor (2)
    (* has higher precedence because grammar is layered *)

  AMBIGUOUS GRAMMAR: one string has two parse trees.
    "if E then if E then S else S" -- dangling else problem
    Fix: introduce matched/unmatched non-terminals (Pascal/C approach)
    Or: precedence climbing / Pratt parser

  OPERATOR PRECEDENCE in C (highest to lowest, selected):
    1.  ()  []  ->  .          postfix
    2.  !  ~  ++  --  (cast)   unary right-to-left
    3.  *  /  %                multiplicative
    4.  +  -                   additive
    5.  <<  >>                 shift
    6.  <  <=  >  >=           relational
    7.  ==  !=                 equality
    8.  &                      bitwise AND
    9.  ^                      bitwise XOR
    10. |                      bitwise OR
    11. &&                     logical AND
    12. ||                     logical OR
    13. ?:                     ternary
    14. =  +=  -=  etc         assignment right-to-left
    15. ,                      comma
""")

# ------------------------------------------------------------------ #
# S2: REGEX THEORY
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 2: REGEX THEORY")
print(SEP)

print("""
  REGULAR EXPRESSION OPERATORS:
    .          any character except newline
    ^          start of string (or line in MULTILINE)
    $          end of string
    *          0 or more (greedy)
    +          1 or more (greedy)
    ?          0 or 1 (makes * + ? lazy when appended: *?)
    {n}        exactly n
    {n,m}      n to m
    [abc]      character class: a or b or c
    [^abc]     negated class: NOT a, b, c
    [a-z]      range
    |          alternation: A|B
    (...)      capturing group
    (?:...)    non-capturing group
    (?=...)    lookahead (zero-width, positive)
    (?!...)    lookahead (zero-width, negative)
    (?<=...)   lookbehind (positive)
    (?<!...)   lookbehind (negative)

  CHARACTER CLASSES:
    \\d   [0-9]
    \\D   [^0-9]
    \\w   [a-zA-Z0-9_]
    \\W   [^\\w]
    \\s   [ \\t\\n\\r\\f\\v]
    \\S   [^\\s]
    \\b   word boundary (zero-width)

  NFA vs DFA:
    Regex -> NFA (Thompson construction, O(n) states)
    NFA -> DFA (subset construction, up to 2^n states)
    Python re module: uses NFA-based backtracking engine
    -> catastrophic backtracking possible on evil patterns:
       (a+)+ on "aaaaaab" -> exponential -> use atomic groups or re2
    re2 / Google RE2: DFA-based, O(n) guaranteed, no backtracking
""")

# ------------------------------------------------------------------ #
# S3: VALIDATION PATTERNS
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 3: INPUT VALIDATION -- NAME, PHONE, PASSWORD")
print(SEP)

# --- FIRST NAME ---
NAME_RE = re.compile(
    r"^[A-Za-z]"           # must start with a letter
    r"[A-Za-z\-' ]{0,48}"  # letters, hyphen, apostrophe, space (Mary-Jane, O'Brien)
    r"[A-Za-z]$"           # must end with a letter (min 2 chars total)
)

name_tests = [
    ("Alice",        True),
    ("O'Brien",      True),
    ("Mary-Jane",    True),
    ("Jean Pierre",  True),
    ("A",            False),   # too short (single char fails end-anchor)
    ("Al1ce",        False),   # digit
    ("-Alice",       False),   # starts with hyphen
    ("Alice-",       False),   # ends with hyphen
    ("A" * 51,       False),   # too long
    ("",             False),
]

print("  FIRST NAME regex:  ^[A-Za-z][A-Za-z\\-' ]{0,48}[A-Za-z]$")
print(f"  {'Input':<20} {'Expected':>10} {'Got':>10} {'Pass':>6}")
print("  " + "-" * 48)
all_pass = True
for val, expected in name_tests:
    got = bool(NAME_RE.match(val))
    ok = got == expected
    if not ok: all_pass = False
    display = (val[:17] + "...") if len(val) > 20 else val
    print(f"  {display:<20} {str(expected):>10} {str(got):>10} {'OK' if ok else 'FAIL':>6}")
print(f"  All name tests: {'PASS' if all_pass else 'FAIL'}")

# --- PHONE NUMBER ---
# accepts: +1-800-555-1234  (800) 555-1234  8005551234  +44 7911 123456
PHONE_RE = re.compile(
    r"^"
    r"(\+\d{1,3}[\s\-]?)?"        # optional country code: +1, +44, +353
    r"(\(?\d{3}\)?[\s\-\.])?"     # optional area code: (800) or 800-
    r"\d{3}"                       # first 3 digits
    r"[\s\-\.]"                    # separator
    r"\d{4}"                       # last 4 digits
    r"$"
)

phone_tests = [
    ("+1-800-555-1234",   True),
    ("(800) 555-1234",    True),
    ("800-555-1234",      True),
    ("8005551234",        False),  # no separator in last group
    ("555-1234",          True),   # local 7-digit
    ("+44 7911 123456",   False),  # UK format, different structure
    ("(800)555-1234",     False),  # missing space after area code
    ("555-123",           False),  # too short
    ("555-12345",         False),  # too long
]

print()
print("  PHONE regex (US-centric): ^(+CCC)?(AAA)?DDD-DDDD$")
print(f"  {'Input':<22} {'Expected':>10} {'Got':>10} {'Pass':>6}")
print("  " + "-" * 50)
all_pass = True
for val, expected in phone_tests:
    got = bool(PHONE_RE.match(val))
    ok = got == expected
    if not ok: all_pass = False
    print(f"  {val:<22} {str(expected):>10} {str(got):>10} {'OK' if ok else 'FAIL':>6}")
print(f"  All phone tests: {'PASS' if all_pass else 'FAIL'}")

# --- PASSWORD STRENGTH ---
# Rules: 8-64 chars, 1 upper, 1 lower, 1 digit, 1 special
# Using lookaheads -- each is a zero-width check from position 0

PWD_RE = re.compile(
    r"^"
    r"(?=.*[A-Z])"          # at least one uppercase
    r"(?=.*[a-z])"          # at least one lowercase
    r"(?=.*\d)"             # at least one digit
    r"(?=.*[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?])"  # at least one special
    r".{8,64}"              # 8 to 64 chars total
    r"$"
)

def password_strength(pwd):
    score = 0
    reasons = []
    if len(pwd) >= 8:  score += 1
    else: reasons.append("too short (<8)")
    if len(pwd) >= 12: score += 1
    if re.search(r"[A-Z]", pwd): score += 1
    else: reasons.append("no uppercase")
    if re.search(r"[a-z]", pwd): score += 1
    else: reasons.append("no lowercase")
    if re.search(r"\d", pwd):    score += 1
    else: reasons.append("no digit")
    if re.search(r"[!@#$%^&*()\-_=+\[\]{};':\"\\|,.<>/?]", pwd): score += 1
    else: reasons.append("no special char")
    if not re.search(r"(.)\1{2,}", pwd): score += 1  # no 3+ repeated chars
    else: reasons.append("repeated chars (aaa)")
    levels = ["Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong", "Excellent"]
    label = levels[min(score, 6)]
    valid = bool(PWD_RE.match(pwd))
    return valid, score, label, reasons

pwd_tests = [
    "password",
    "Password1",
    "P@ssw0rd",
    "Tr0ub4dor&3",
    "correcthorsebatterystaple",
    "C0rrect!Horse#Battery",
    "aaaaAAAA1!",
    "short1A!",
]

print()
print("  PASSWORD validation (8-64, upper+lower+digit+special):")
print(f"  {'Password':<26} {'Valid':>6} {'Score':>6} {'Level':<14} Issues")
print("  " + "-" * 70)
for pwd in pwd_tests:
    valid, score, label, reasons = password_strength(pwd)
    issues = ", ".join(reasons) if reasons else "none"
    print(f"  {pwd:<26} {str(valid):>6} {score:>6} {label:<14} {issues}")

# ------------------------------------------------------------------ #
# S4: C HEADER  validation.h
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: validation.h  (C header -- POSIX regex.h)")
print(SEP)

c_header = r"""
/* validation.h
 * Input validation using POSIX extended regex (regex.h).
 * Compile:  gcc main.c -o main
 * Include:  #include "validation.h"
 */
#ifndef VALIDATION_H
#define VALIDATION_H

#include <regex.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ---- patterns ---- */
#define RE_FIRST_NAME  "^[A-Za-z][A-Za-z' -]{0,48}[A-Za-z]$"
#define RE_PHONE_US    "^(\\+[0-9]{1,3}[- ]?)?(\\(?[0-9]{3}\\)?[- .])?" \
                       "[0-9]{3}[- .][0-9]{4}$"

/* password rules checked individually (lookaheads not in POSIX) */
#define RE_PWD_UPPER   "[A-Z]"
#define RE_PWD_LOWER   "[a-z]"
#define RE_PWD_DIGIT   "[0-9]"
#define RE_PWD_SPECIAL "[!@#$%^&*()+\\-={}\\[\\];':\"\\\\|,.<>/?]"

typedef enum {
    VALID   = 0,
    ERR_TOO_SHORT,
    ERR_NO_UPPER,
    ERR_NO_LOWER,
    ERR_NO_DIGIT,
    ERR_NO_SPECIAL,
    ERR_REGEX_FAIL,
} valid_t;

/* returns VALID or error code */
static inline valid_t validate_name(const char *name) {
    regex_t re;
    int r;
    if (regcomp(&re, RE_FIRST_NAME, REG_EXTENDED | REG_NOSUB) != 0)
        return ERR_REGEX_FAIL;
    r = regexec(&re, name, 0, NULL, 0);
    regfree(&re);
    return (r == 0) ? VALID : ERR_REGEX_FAIL;
}

static inline valid_t validate_phone(const char *phone) {
    regex_t re;
    int r;
    if (regcomp(&re, RE_PHONE_US, REG_EXTENDED | REG_NOSUB) != 0)
        return ERR_REGEX_FAIL;
    r = regexec(&re, phone, 0, NULL, 0);
    regfree(&re);
    return (r == 0) ? VALID : ERR_REGEX_FAIL;
}

static inline int _re_match(const char *pat, const char *s) {
    regex_t re;
    int r;
    if (regcomp(&re, pat, REG_EXTENDED | REG_NOSUB) != 0) return 0;
    r = regexec(&re, s, 0, NULL, 0);
    regfree(&re);
    return r == 0;
}

static inline valid_t validate_password(const char *pwd) {
    size_t len = strlen(pwd);
    if (len < 8 || len > 64) return ERR_TOO_SHORT;
    if (!_re_match(RE_PWD_UPPER,   pwd)) return ERR_NO_UPPER;
    if (!_re_match(RE_PWD_LOWER,   pwd)) return ERR_NO_LOWER;
    if (!_re_match(RE_PWD_DIGIT,   pwd)) return ERR_NO_DIGIT;
    if (!_re_match(RE_PWD_SPECIAL, pwd)) return ERR_NO_SPECIAL;
    return VALID;
}

static inline const char *valid_str(valid_t v) {
    switch (v) {
        case VALID:           return "VALID";
        case ERR_TOO_SHORT:   return "ERROR: too short or too long (8-64)";
        case ERR_NO_UPPER:    return "ERROR: need at least one uppercase letter";
        case ERR_NO_LOWER:    return "ERROR: need at least one lowercase letter";
        case ERR_NO_DIGIT:    return "ERROR: need at least one digit";
        case ERR_NO_SPECIAL:  return "ERROR: need at least one special character";
        case ERR_REGEX_FAIL:  return "ERROR: invalid format";
        default:              return "ERROR: unknown";
    }
}

#endif /* VALIDATION_H */
"""
print(c_header)

# ------------------------------------------------------------------ #
# S5: TERMINAL UI IN C (ncurses sketch)
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 5: TERMINAL UI IN C  (ncurses form sketch)")
print(SEP)

c_ui = r"""
/* form_ui.c  -- ncurses input form: first name + phone + password
 * Build:  gcc form_ui.c -lncurses -o form_ui
 */
#include <ncurses.h>
#include <string.h>
#include "validation.h"

#define MAX_FIELD 65

typedef struct {
    char first_name[MAX_FIELD];
    char phone[MAX_FIELD];
    char password[MAX_FIELD];
} form_data_t;

/* draw a labeled field, return input string */
static void draw_field(int row, const char *label, char *buf,
                       int maxlen, int hide) {
    mvprintw(row, 2, "%-16s", label);
    /* input box */
    mvaddch(row, 19, '[');
    mvaddch(row, 19 + maxlen + 1, ']');
    move(row, 20);

    int i = 0;
    int ch;
    while (i < maxlen - 1) {
        ch = getch();
        if (ch == '\n' || ch == KEY_ENTER) break;
        if ((ch == KEY_BACKSPACE || ch == 127) && i > 0) {
            i--;
            mvaddch(row, 20 + i, ' ');
            move(row, 20 + i);
            continue;
        }
        if (ch >= 32 && ch < 127) {
            buf[i++] = (char)ch;
            addch(hide ? '*' : (chtype)ch);
        }
    }
    buf[i] = '\0';
}

int main(void) {
    form_data_t data = {0};

    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);

    /* title */
    attron(A_BOLD);
    mvprintw(1, 2, "=== User Registration ===");
    attroff(A_BOLD);

    /* fields */
    draw_field(4,  "First Name:",  data.first_name, 50, 0);
    draw_field(6,  "Phone:",       data.phone,       20, 0);
    draw_field(8,  "Password:",    data.password,    64, 1);

    /* validate */
    valid_t vn = validate_name(data.first_name);
    valid_t vp = validate_phone(data.phone);
    valid_t vw = validate_password(data.password);

    /* show results */
    mvprintw(11, 2, "Name:     %s", valid_str(vn));
    mvprintw(12, 2, "Phone:    %s", valid_str(vp));
    mvprintw(13, 2, "Password: %s", valid_str(vw));

    if (vn == VALID && vp == VALID && vw == VALID) {
        attron(A_BOLD | COLOR_PAIR(1));
        mvprintw(15, 2, "[ Registration OK ]");
        attroff(A_BOLD | COLOR_PAIR(1));
    } else {
        mvprintw(15, 2, "[ Fix errors above, press any key ]");
    }

    mvprintw(17, 2, "Press any key to exit...");
    getch();
    endwin();
    return 0;
}
/* NOTES:
 *   - ncurses handles terminal raw mode, cursor, color pairs
 *   - validation.h is stateless: no malloc, pure POSIX regex
 *   - password field echoes '*' (hide=1 in draw_field)
 *   - for color:  start_color(); init_pair(1, COLOR_GREEN, COLOR_BLACK);
 *   - link with -lncurses  (sudo apt install libncurses-dev on Debian)
 */
"""
print(c_ui)

# ------------------------------------------------------------------ #
# S6: GRAMMAR SYNTAX QUICK REFERENCE
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 6: QUICK REFERENCE")
print(SEP)
print(r"""
  REGEX METACHAR   MEANING                    EXAMPLE
  ------------------------------------------------------------
  .                any char (not newline)     a.c -> abc, aXc
  ^  $             start / end of string      ^abc$ = exact match
  *  +  ?          0+  1+  0or1               col+r -> color, colour
  {n,m}            n to m repetitions         \d{3,4} = 3 or 4 digits
  [A-Za-z]         character class / range    [aeiou] = vowel
  [^...]           negated class              [^\d] = non-digit
  (abc)            capturing group            (foo|bar) = foo or bar
  (?:abc)          non-capturing group        (?:foo|bar)
  (?=...)          positive lookahead         \d(?=px) = digit before px
  (?!...)          negative lookahead         \d(?!px) = digit NOT before px
  \b               word boundary              \bword\b = exact word

  GRAMMAR NOTATION  MEANING
  ------------------------------------------------------------
  <S> ::= ...       BNF production (::= means "derives to")
  { ... }           EBNF: zero or more repetitions
  [ ... ]           EBNF: optional (zero or one)
  ( ... )           EBNF: grouping
  |                 alternation (either/or)
  "..."             terminal literal

  VALIDATION SUMMARY:
  Field        Pattern key                        Notes
  ---------------------------------------------------------------
  First name   ^[A-Za-z][A-Za-z' -]{0,48}[A-Za-z]$  apostrophe, hyphen OK
  US phone     (CCC)?-(AAA)?-DDD-DDDD              +1-800-555-1234
  Password     lookahead x4: upper lower digit special  8-64 chars
""")

print(SEP)
print("Done.")
print(SEP)
