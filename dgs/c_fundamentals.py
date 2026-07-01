"""C programming fundamentals -- data types, operators, syntax, interview patterns.

WHY C FOR A JOB:
  C is the language of the machine. Every operating system kernel, embedded system,
  high-performance library (NumPy, PyTorch C++ backend, CUDA runtime) is C or C++.
  An EE/physics grad who knows C can:
  - Read datasheets and talk to hardware (SPI/I2C via ioctl)
  - Write firmware (microcontrollers: Raspberry Pi, STM32, Arduino)
  - Understand what Python is actually doing underneath
  - Pass whiteboard interviews: bit manipulation, pointers, memory layout

THE NATURAL PHILOSOPHER PROBLEM:
  Pre-1800, "natural philosopher" meant what we call "scientist" or "engineer" today.
  Newton, Faraday, Maxwell were natural philosophers -- they did math, physics,
  experiment, and engineering all at once. C is what the modern natural philosopher
  uses when they need to talk to hardware. Python is for prototyping; C is for shipping.

WHAT TO MEMORIZE FOR AN OFFICE JOB (the minimal set):
  1. Data types and sizes (the sizeof table below)
  2. Bitwise operators: &, |, ^, ~, <<, >> (used in every embedded/networking job)
  3. Pointer syntax: int *p = &x; *p = 5;  (the dereferencing paradigm)
  4. Array indexing: a[i] == *(a + i) (pointer arithmetic)
  5. Struct layout and alignment
  6. Common patterns: bit test, bit set, bit clear, count bits, is power of 2

ROT CIPHER (also called rotation cipher):
  ROT13: shift each letter by 13 positions. A->N, B->O, Z->M.
  ROT13 is its own inverse: ROT13(ROT13(x)) = x.
  In C, this is: char rot13(char c) { ... }
  In bitwise terms: ROT (bit rotation) is different -- circular shift:
    rotl(x, n) = (x << n) | (x >> (32 - n))  // left rotate 32-bit integer
  Both are interview questions. ROT13 is string manipulation; bit rotation is hardware.

THE &| vs && || DISTINCTION (common syntax error for beginners):
  &   = bitwise AND (operates on individual bits of integers)
  |   = bitwise OR
  &&  = logical AND (short-circuit: evaluates right side only if left is true)
  ||  = logical OR (short-circuit: evaluates right side only if left is false)

  DANGER: if (a & b)  -- tests if any bit is set in common (BITWISE)
           if (a && b) -- tests if BOTH are truthy (LOGICAL) -- what you usually want

  EXAMPLE: a=2 (0b10), b=1 (0b01)
    a & b  = 0 (no bits in common) -> if(a & b) is FALSE
    a && b = 1 (both nonzero)      -> if(a && b) is TRUE
"""
import ctypes
import struct
import sympy as sp


# ── C data types and sizes ────────────────────────────────────────────

C_TYPES = [
    # (name, ctypes_type, size_bytes, signed, format_specifier, notes)
    ("char",           ctypes.c_char,   1, "signed",   "%c",   "character; also int8 range -128..127"),
    ("unsigned char",  ctypes.c_ubyte,  1, "unsigned", "%u",   "uint8 range 0..255"),
    ("short",          ctypes.c_short,  2, "signed",   "%hd",  "int16 range -32768..32767"),
    ("unsigned short", ctypes.c_ushort, 2, "unsigned", "%hu",  "uint16 range 0..65535"),
    ("int",            ctypes.c_int,    4, "signed",   "%d",   "int32 range -2B..+2B"),
    ("unsigned int",   ctypes.c_uint,   4, "unsigned", "%u",   "uint32 range 0..4B"),
    ("long",           ctypes.c_long,   8, "signed",   "%ld",  "int64 on Linux; int32 on Windows!"),
    ("long long",      ctypes.c_longlong, 8, "signed", "%lld", "int64 guaranteed"),
    ("unsigned long long", ctypes.c_ulonglong, 8, "unsigned", "%llu", "uint64"),
    ("float",          ctypes.c_float,  4, "float",    "%f",   "IEEE 754 single ~7 sig digits"),
    ("double",         ctypes.c_double, 8, "float",    "%lf",  "IEEE 754 double ~15 sig digits"),
    ("pointer",        ctypes.c_void_p, 8, "address",  "%p",   "64-bit on x86-64; 4 bytes on 32-bit"),
    ("size_t",         ctypes.c_size_t, 8, "unsigned", "%zu",  "result of sizeof; always unsigned"),
]


def c_type_table():
    """Return C type sizes and properties as a list of dicts."""
    rows = []
    for entry in C_TYPES:
        name, ctype, size, sign, fmt, note = entry
        try:
            actual_size = ctypes.sizeof(ctype)
        except Exception:
            actual_size = size
        rows.append({
            "type": name,
            "size_bytes": actual_size,
            "size_bits": actual_size * 8,
            "signedness": sign,
            "printf_format": fmt,
            "notes": note,
        })
    return rows


def sizeof_check(type_name):
    """Get the size of a C type by name."""
    lookup = {r["type"]: r for r in c_type_table()}
    if type_name not in lookup:
        available = list(lookup.keys())
        raise ValueError(f"Unknown type '{type_name}'. Available: {available}")
    return lookup[type_name]


# ── bitwise operators ─────────────────────────────────────────────────

def bitwise_demo(a, b, n_bits=8):
    """Demonstrate all bitwise operators on integers a and b.

    These are the operators you MUST know for any low-level C job:
      &  = AND  (mask bits)
      |  = OR   (set bits)
      ^  = XOR  (toggle bits, detect differences, no-temp swap)
      ~  = NOT  (flip all bits; note: ~a = -(a+1) for signed)
      << = left shift  (multiply by 2^n, fast)
      >> = right shift (divide by 2^n for unsigned; arithmetic for signed)
    """
    mask = (1 << n_bits) - 1
    a_bits = f"{a & mask:0{n_bits}b}"
    b_bits = f"{b & mask:0{n_bits}b}"
    return {
        "a": a, "b": b,
        "a_binary": a_bits, "b_binary": b_bits,
        "AND":  {"value": a & b,      "binary": f"{(a & b) & mask:0{n_bits}b}", "use": "mask -- keep only bits set in BOTH"},
        "OR":   {"value": a | b,      "binary": f"{(a | b) & mask:0{n_bits}b}", "use": "set bits -- turn ON any bit set in either"},
        "XOR":  {"value": a ^ b,      "binary": f"{(a ^ b) & mask:0{n_bits}b}", "use": "toggle -- flip bits where they differ"},
        "NOT_a":{"value": ~a & mask,  "binary": f"{~a & mask:0{n_bits}b}",      "use": "invert -- flip all bits (signed: ~a = -(a+1))"},
        "SHL2": {"value": (a << 2) & mask, "binary": f"{(a << 2) & mask:0{n_bits}b}", "use": "left shift by 2 = multiply by 4"},
        "SHR1": {"value": a >> 1,     "binary": f"{(a >> 1) & mask:0{n_bits}b}", "use": "right shift by 1 = divide by 2"},
        "logical_AND": bool(a) and bool(b),   # && in C
        "logical_OR":  bool(a) or bool(b),    # || in C
        "danger_note": f"a={a}(0b{a_bits}), b={b}(0b{b_bits}): a&b={a&b} vs a&&b={bool(a and b)}",
    }


# ── common bit manipulation patterns (interview gold) ─────────────────

def bit_count(n):
    """Count number of 1-bits in integer n (popcount / Hamming weight).

    Method 1 (Kernighan): n &= (n-1) clears the lowest set bit.
    O(k) where k = number of set bits.

    Method 2: built-in bin(n).count('1')
    In C: __builtin_popcount(n) (GCC intrinsic, compiles to POPCNT instruction).
    """
    if n < 0:
        raise ValueError("bit_count expects non-negative integer")
    # Kernighan's bit clearing method
    count = 0
    x = n
    while x:
        x &= (x - 1)
        count += 1
    return {"n": n, "binary": bin(n), "popcount": count,
            "kernighan_steps": count,
            "c_intrinsic": "__builtin_popcount(n)"}


def is_power_of_2(n):
    """Test if n is a power of 2 using the n & (n-1) == 0 trick.

    Powers of 2 in binary: 1=0b001, 2=0b010, 4=0b100, 8=0b1000
    n-1 flips all lower bits: 4-1=3=0b011; 4&3=0b000=0 -> is power of 2.
    NOT a power of 2: n=6=0b110; n-1=5=0b101; 6&5=0b100 != 0.

    EDGE CASE: n=0 is NOT a power of 2 (0 & -1 = 0 would give wrong answer).
    """
    if n <= 0:
        return {"n": n, "is_power_of_2": False, "note": "0 and negatives are not powers of 2"}
    result = (n & (n - 1)) == 0
    return {"n": n, "binary": bin(n), "is_power_of_2": result,
            "trick": "n & (n-1) == 0"}


def bit_set(n, pos):
    """Set bit at position pos: n | (1 << pos)."""
    return {"result": n | (1 << pos), "binary": bin(n | (1 << pos)),
            "c_code": f"n | (1 << {pos})"}


def bit_clear(n, pos):
    """Clear bit at position pos: n & ~(1 << pos)."""
    return {"result": n & ~(1 << pos), "binary": bin(n & ~(1 << pos)),
            "c_code": f"n & ~(1 << {pos})"}


def bit_toggle(n, pos):
    """Toggle bit at position pos: n ^ (1 << pos)."""
    return {"result": n ^ (1 << pos), "binary": bin(n ^ (1 << pos)),
            "c_code": f"n ^ (1 << {pos})"}


def bit_test(n, pos):
    """Test if bit at position pos is set: (n >> pos) & 1."""
    result = bool((n >> pos) & 1)
    return {"result": result, "c_code": f"(n >> {pos}) & 1"}


def rotate_left(n, shift, bits=32):
    """Circular left rotation of an n-bit integer.

    rotl(x, shift) = (x << shift) | (x >> (bits - shift))
    Used in cryptography (SHA, MD5, AES), hardware CRC, RISC-V ROR instruction.
    NOT the same as ROT13 (which rotates alphabet characters).
    """
    if bits not in (8, 16, 32, 64):
        raise ValueError("bits must be 8, 16, 32, or 64")
    mask = (1 << bits) - 1
    shift = shift % bits
    result = ((n << shift) | (n >> (bits - shift))) & mask
    return {"input": n, "input_binary": f"{n & mask:0{bits}b}",
            "shift": shift, "result": result,
            "result_binary": f"{result:0{bits}b}",
            "c_code": f"(x << {shift}) | (x >> ({bits} - {shift}))"}


def no_temp_swap(a, b):
    """Swap two integers without a temporary variable using XOR.

    a ^= b    (a = a XOR b)
    b ^= a    (b = b XOR (a XOR b) = original a)
    a ^= b    (a = (a XOR b) XOR original_a = original b)

    Why it works: XOR is its own inverse: x^y^y = x.
    Caveat: FAILS if a and b are the same memory location (a == b: gives 0).
    """
    x, y = a, b
    x ^= y
    y ^= x
    x ^= y
    return {"original": (a, b), "after_swap": (x, y),
            "c_code": "a^=b; b^=a; a^=b;",
            "warning": "FAILS if &a == &b (aliasing -- same variable)"}


# ── ROT ciphers ───────────────────────────────────────────────────────

def rot13(text):
    """ROT13 cipher: shift each letter by 13. ROT13(ROT13(x)) = x.

    In C:
      char rot13(char c) {
        if (c >= 'a' && c <= 'z') return 'a' + (c - 'a' + 13) % 26;
        if (c >= 'A' && c <= 'Z') return 'A' + (c - 'A' + 13) % 26;
        return c;
      }
    """
    result = []
    for c in text:
        if 'a' <= c <= 'z':
            result.append(chr((ord(c) - ord('a') + 13) % 26 + ord('a')))
        elif 'A' <= c <= 'Z':
            result.append(chr((ord(c) - ord('A') + 13) % 26 + ord('A')))
        else:
            result.append(c)
    encoded = ''.join(result)
    return {"input": text, "encoded": encoded,
            "is_self_inverse": rot13.__name__,
            "verify": ''.join(
                chr((ord(c) - ord('a') + 13) % 26 + ord('a')) if 'a' <= c <= 'z'
                else chr((ord(c) - ord('A') + 13) % 26 + ord('A')) if 'A' <= c <= 'Z'
                else c for c in encoded) == text}


def rot_n(text, n=13):
    """ROT-n cipher (generalization of ROT13 to any shift n)."""
    n = n % 26
    result = []
    for c in text:
        if 'a' <= c <= 'z':
            result.append(chr((ord(c) - ord('a') + n) % 26 + ord('a')))
        elif 'A' <= c <= 'Z':
            result.append(chr((ord(c) - ord('A') + n) % 26 + ord('A')))
        else:
            result.append(c)
    return {"input": text, "n": n, "encoded": ''.join(result),
            "decode_with": f"rot_n(encoded, {26-n})"}


# ── common syntax errors (beginner traps) ─────────────────────────────

SYNTAX_ERRORS = [
    {
        "error": "Assignment in condition",
        "wrong":   "if (x = 5)   // assigns 5 to x, always true (non-zero)",
        "correct": "if (x == 5)  // comparison",
        "c_trick": "Some write: if (5 == x) -- compiler error if you typo '=' (Yoda condition)",
    },
    {
        "error": "Missing break in switch",
        "wrong":   "case 1: do_thing();  case 2: ...  // falls through!",
        "correct": "case 1: do_thing(); break;  case 2: ...",
        "c_trick": "Intentional fallthrough: add comment /* FALLTHROUGH */",
    },
    {
        "error": "Integer division truncation",
        "wrong":   "float x = 5 / 2;   // x = 2.0, not 2.5",
        "correct": "float x = 5.0 / 2; // or (float)5 / 2",
        "c_trick": "If numerator and denominator are both int, result is int",
    },
    {
        "error": "Off-by-one in array",
        "wrong":   "int a[10]; a[10] = 5;  // writes PAST the array (UB)",
        "correct": "a[9] = 5;  // last valid index is N-1",
        "c_trick": "Arrays: 0-indexed, size N, valid indices 0..N-1",
    },
    {
        "error": "Forgetting to dereference pointer",
        "wrong":   "int *p = &x; p = 5;   // sets the POINTER to 0x5, not x",
        "correct": "*p = 5;  // dereference: writes 5 into x",
        "c_trick": "int *p: 'p' is the address, '*p' is the value at that address",
    },
    {
        "error": "String not null-terminated",
        "wrong":   "char s[5] = {'H','e','l','l','o'};  // no '\\0' -- strcmp/printf crash",
        "correct": "char s[6] = \"Hello\";  // compiler adds '\\0' automatically",
        "c_trick": "Always allocate strlen(s)+1 bytes for a C string",
    },
    {
        "error": "Using freed memory",
        "wrong":   "free(p); *p = 5;  // use-after-free: undefined behavior",
        "correct": "free(p); p = NULL;  // null the pointer after free",
        "c_trick": "Set pointer to NULL after free; dereferencing NULL -> segfault (detectable)",
    },
    {
        "error": "Bitwise vs logical AND/OR",
        "wrong":   "if (a & b)  // true only if they share a set bit",
        "correct": "if (a && b) // true if both are nonzero",
        "c_trick": "a=2(0b10), b=1(0b01): a&b=0 (FALSE), a&&b=1 (TRUE)",
    },
    {
        "error": "sizeof pointer vs array",
        "wrong":   "void f(int a[]) { int n = sizeof(a)/sizeof(a[0]); } // n=2 not length!",
        "correct": "pass array length as a separate parameter",
        "c_trick": "Inside a function, array decays to pointer; sizeof(a)=8 (pointer size)",
    },
]


def syntax_errors_quiz():
    """Return the list of common C syntax errors as a study guide."""
    return SYNTAX_ERRORS


# ── pointer primer ────────────────────────────────────────────────────

POINTER_CHEATSHEET = {
    "declare": "int *p;          // p is a pointer to int",
    "init":    "int x = 5; int *p = &x;  // & = address-of operator",
    "deref":   "*p = 10;         // write 10 into x via pointer",
    "read":    "int y = *p;      // read the value at address p",
    "null":    "int *p = NULL;   // null pointer; dereferencing is UB/segfault",
    "array":   "int a[5]; int *p = a;  // array name IS a pointer to a[0]",
    "arith":   "*(p + 2)         // same as a[2] -- pointer arithmetic",
    "func":    "void f(int *p) { *p = 5; }  // pass by pointer -> modifies original",
    "const1":  "const int *p     // p points to const int -- *p cannot change",
    "const2":  "int * const p    // const pointer -- p cannot change, *p can",
    "double":  "int **pp         // pointer to pointer -- used for 2D arrays, argv",
    "sizeof":  "sizeof(*p)       // size of the type pointed to (4 for int*)",
    "malloc":  "int *p = malloc(10 * sizeof(int));  // heap allocation",
    "free":    "free(p); p = NULL;  // always null after free",
}


def pointer_cheatsheet():
    """Return C pointer syntax cheatsheet."""
    return POINTER_CHEATSHEET


# ── struct layout ─────────────────────────────────────────────────────

STRUCT_DEMO = """
// struct in C -- padding and alignment matter for sizeof()
struct Padded {
    char   a;    // 1 byte
    // 3 bytes PADDING (compiler aligns int to 4-byte boundary)
    int    b;    // 4 bytes
    char   c;    // 1 byte
    // 3 bytes PADDING at end (struct size must be multiple of largest member)
    // total sizeof(struct Padded) = 12 bytes, NOT 6
};

struct Packed {
    int    b;    // 4 bytes (put largest first)
    char   a;    // 1 byte
    char   c;    // 1 byte
    // 2 bytes PADDING at end
    // total sizeof(struct Packed) = 8 bytes (better!)
};

// Force no padding: __attribute__((packed))  or  #pragma pack(1)
// But misaligned access is SLOW on x86 and CRASH on ARM

// LESSON: order struct members by decreasing size to minimize padding
"""


def struct_layout_lesson():
    """Return the struct alignment rules as a string."""
    return {
        "rule": "Order members by DECREASING size to minimize padding",
        "alignment_rule": "Each member aligns to its own size (int at 4-byte, double at 8-byte)",
        "sizeof_padded": 12,
        "sizeof_packed": 8,
        "demo_c_code": STRUCT_DEMO,
        "pragma": "__attribute__((packed)) forces 1-byte alignment (GCC/Clang)",
    }


# ── interview question: reverse bits ─────────────────────────────────

def reverse_bits(n, n_bits=32):
    """Reverse the bits of a 32-bit integer.

    Naive O(n_bits) approach: take bits from LSB and put them in MSB.
    In C: while (n) { result = (result << 1) | (n & 1); n >>= 1; count++; }
    Fast: lookup table or bit interleaving tricks.
    """
    if n < 0:
        raise ValueError("reverse_bits expects non-negative integer")
    mask = (1 << n_bits) - 1
    n &= mask
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return {
        "input": n,
        "result": result,
        "input_binary": f"{n:0{n_bits}b}",
        "result_binary": f"{result:0{n_bits}b}",
        "c_code": "// while(n){result=(result<<1)|(n&1); n>>=1;}",
    }


# ── study plan for C job interview ────────────────────────────────────

def c_interview_study_plan():
    """Priority-ordered study plan for C job interview prep."""
    return [
        {"priority": 1, "topic": "Data types and sizeof",
         "practice": "Write char/int/long/pointer sizes from memory; no lookup"},
        {"priority": 2, "topic": "Bitwise operators & | ^ ~ << >>",
         "practice": "Trace: a=0b1010, b=0b1100: AND, OR, XOR by hand"},
        {"priority": 3, "topic": "Pointer syntax: *, &, ->, NULL",
         "practice": "Write a swap function using pointers; trace memory"},
        {"priority": 4, "topic": "Array as pointer; pointer arithmetic",
         "practice": "Write strlen(s) using pointer arithmetic (no index)"},
        {"priority": 5, "topic": "Bit manipulation patterns",
         "practice": "is_power_of_2, bit_count, set/clear/toggle/test by heart"},
        {"priority": 6, "topic": "String functions: strlen, strcpy, strcat, strcmp",
         "practice": "Implement strlen without library; trace buffer overflow"},
        {"priority": 7, "topic": "Struct layout and alignment",
         "practice": "Predict sizeof for a given struct with mixed types"},
        {"priority": 8, "topic": "malloc/free and memory model",
         "practice": "Write a linked list in C; detect memory leak"},
        {"priority": 9, "topic": "Preprocessor: #define, #ifdef, macros",
         "practice": "Write a MIN(a,b) macro; explain why (a)<(b)?(a):(b) is safe"},
        {"priority": 10, "topic": "Common errors: = vs ==, & vs &&, off-by-one",
         "practice": "Read code with deliberate bugs and find them"},
    ]


# ── SymPy connection: C types as mathematical objects ─────────────────

def c_types_sympy_5():
    """Five key relationships between C types and math, in SymPy."""
    n = sp.Symbol('n', positive=True, integer=True)

    return {
        "int8_range": sp.Eq(sp.Symbol("range_int8"),
                            sp.Interval(-2**7, 2**7 - 1)),
        "uint_range": sp.Eq(sp.Symbol("range_uintN"),
                            sp.Interval(0, 2**n - 1)),
        "float32_precision": sp.Eq(sp.Symbol("epsilon_float32"),
                                   sp.Rational(1, 2**23)),
        "bitmask_n_ones": sp.Eq(sp.Symbol("mask"),
                                2**n - 1),
        "rotate_left_formula": sp.Eq(
            sp.Symbol("rotl_x_n"),
            sp.Symbol("x_shl_n_OR_x_shr_32_minus_n")),
    }


if __name__ == "__main__":
    print("=== C data types and sizes ===")
    for row in c_type_table():
        print(f"  {row['type']:22s} {row['size_bytes']} bytes  "
              f"{row['size_bits']:3d} bits  {row['printf_format']:5s}  {row['notes']}")

    print("\n=== Bitwise operators on a=0b10110 (22), b=0b01101 (13) ===")
    bd = bitwise_demo(22, 13)
    for op in ["AND", "OR", "XOR", "NOT_a", "SHL2", "SHR1"]:
        print(f"  {op:7s}: {bd[op]['binary']}  ({bd[op]['value']:3d})  -- {bd[op]['use']}")
    print(f"  DANGER: {bd['danger_note']}")

    print("\n=== Bit manipulation patterns ===")
    print(f"  bit_count(0b10110101 = 181): {bit_count(181)['popcount']}")
    print(f"  is_power_of_2(64): {is_power_of_2(64)['is_power_of_2']}")
    print(f"  is_power_of_2(63): {is_power_of_2(63)['is_power_of_2']}")
    print(f"  bit_set(0b1010, pos=0):  {bit_set(0b1010, 0)['c_code']} = {bit_set(0b1010, 0)['binary']}")
    print(f"  bit_clear(0b1010, pos=1):{bit_clear(0b1010, 1)['c_code']} = {bit_clear(0b1010, 1)['binary']}")
    print(f"  no_temp_swap(3, 7): {no_temp_swap(3, 7)['after_swap']}")

    print("\n=== ROT13 cipher ===")
    r = rot13("Hello World -- Physics and Engineering!")
    print(f"  input:   {r['input']}")
    print(f"  ROT13:   {r['encoded']}")
    print(f"  ROT13 is self-inverse (verified): {r['verify']}")

    print("\n=== Common syntax errors (top 3) ===")
    for err in SYNTAX_ERRORS[:3]:
        print(f"  [{err['error']}]")
        print(f"    WRONG:   {err['wrong']}")
        print(f"    CORRECT: {err['correct']}")

    print("\n=== C interview study plan (priority order) ===")
    for item in c_interview_study_plan():
        print(f"  P{item['priority']}: {item['topic']}")
        print(f"     -> {item['practice']}")
