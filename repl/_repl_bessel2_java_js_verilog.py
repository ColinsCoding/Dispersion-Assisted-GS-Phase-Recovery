# -*- coding: utf-8 -*-
"""
_repl_bessel2_java_js_verilog.py
=================================
Precise Bessel symmetry + Java + JavaScript + Verilog.

S1: Bessel functions -- precise negative-order symmetry
    J_{-n}(x) = (-1)^n J_n(x)  proof via series
    Y_n singularity at x=0, precise behavior
    Spherical Bessel j_n, y_n
    Modified Bessel I_n, K_n
    Wronskians, asymptotic forms
    Numerical precision: catastrophic cancellation near x=0

S2: Java reference
    Types, OOP (class/interface/abstract), generics
    Collections (List, Map, Set), streams, lambdas
    Exception handling, concurrency basics
    Comparison table: Java vs Python vs C

S3: JavaScript reference
    var/let/const, hoisting, closures
    Prototype chain, this keyword
    Promises, async/await
    ES6+ features: destructuring, spread, arrow functions
    DOM manipulation, event loop
    Node.js module system

S4: Verilog HDL
    Module, port declarations, always blocks
    Combinational vs sequential (reg vs wire)
    Blocking (=) vs non-blocking (<=) assignments
    Testbench structure
    Synthesis implications
    Example: 4-bit ALU in Verilog

Output: repl/_out_bessel2_java_js_verilog.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import special as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import (symbols, sqrt, pi, exp, sin, cos, Rational, oo,
                   besselj, bessely, besseli, besselk,
                   simplify, series, diff, integrate, factorial, gamma,
                   limit, log, I)
import os

OUT = os.path.join(os.path.dirname(__file__), "_out_bessel2_java_js_verilog.png")
SEP = "=" * 65

# ============================================================
# S1: BESSEL FUNCTIONS -- PRECISE SYMMETRY AND NUMERICS
# ============================================================
print(SEP)
print("SECTION 1: BESSEL FUNCTIONS -- PRECISE SYMMETRY")
print(SEP)

x_s = symbols("x", positive=True)
n_s = symbols("n", integer=True, nonnegative=True)

print("""
  BESSEL EQUATION (order nu, real or complex):
    x^2 y'' + x y' + (x^2 - nu^2) y = 0

  FROBENIUS SERIES SOLUTION (Bessel function of 1st kind):
    J_nu(x) = SUM_{m=0}^{inf} (-1)^m / (m! * Gamma(m+nu+1)) * (x/2)^(2m+nu)
    Convergent for ALL x (radius of convergence = infinity).
    J_nu is the ONLY solution finite at x=0 for nu >= 0.

  NEGATIVE INTEGER ORDER -- SYMMETRY PROOF:
    Claim: J_{-n}(x) = (-1)^n * J_n(x)  for integer n.

    Proof:
    J_{-n}(x) = SUM_{m=0}^{inf} (-1)^m / (m! * Gamma(m-n+1)) * (x/2)^(2m-n)
    When m = 0,1,...,n-1: Gamma(m-n+1) = Gamma(non-positive int) = +/-inf
    => those terms VANISH (1/Gamma has zeros at non-positive integers).
    Remaining terms: re-index with k = m-n (so m = k+n):
    J_{-n}(x) = SUM_{k=0}^{inf} (-1)^(k+n) / ((k+n)! * Gamma(k+1)) * (x/2)^(2k+n)
              = (-1)^n * SUM_{k=0}^{inf} (-1)^k / (k! * (k+n)!) * (x/2)^(2k+n)
              = (-1)^n * J_n(x)  QED

    CONSEQUENCE: J_{-n} and J_n are LINEARLY DEPENDENT for integer n.
    => Need Y_n (Neumann function) as second independent solution.
    => For NON-INTEGER nu: J_nu and J_{-nu} are independent (use both).
""")

# SymPy verification
print("  SymPy verification of J_{-n} = (-1)^n * J_n(x) for n=0,1,2,3:")
x_num = 2.5
for n in range(4):
    Jp = float(sc.jv(n, x_num))
    Jn = float(sc.jv(-n, x_num))
    sign = (-1)**n
    print(f"    n={n}: J_{n}({x_num})={Jp:.6f},  "
          f"J_{{-{n}}}({x_num})={Jn:.6f},  "
          f"(-1)^n*J_n={sign*Jp:.6f},  match={abs(Jn - sign*Jp) < 1e-10}")

print("""
  BESSEL FUNCTION OF 2ND KIND (Neumann function) Y_n(x):
    Y_0(x) = (2/pi) * [ln(x/2) + gamma_EM] * J_0(x)
             + (2/pi) * SUM_{m=1}^{inf} (-1)^(m+1) * H_m / (m!)^2 * (x/2)^(2m)
    where gamma_EM = 0.5772156... (Euler-Mascheroni constant)
    H_m = 1 + 1/2 + 1/3 + ... + 1/m (harmonic number)

    SINGULAR AT x=0:
    Y_0(x) ~ (2/pi) * ln(x)  as x -> 0
    Y_n(x) ~ -(n-1)! / pi * (2/x)^n  as x -> 0  for n >= 1

    PHYSICAL MEANING:
    Y_n diverges at the origin => exclude from solutions that include x=0.
    For ANNULAR region (r_inner < r < r_outer): KEEP Y_n.
    For FULL disk (r=0 included): SET coefficient of Y_n to ZERO.

  NEGATIVE-ORDER Y:
    Y_{-n}(x) = (-1)^n * Y_n(x)  for integer n.
    (Same symmetry as J, derived from Y_nu definition.)
""")

# Numerical check Y symmetry
print("  Y_{-n} = (-1)^n * Y_n verification:")
for n in [1, 2, 3]:
    Yp = float(sc.yv(n, x_num))
    Yn = float(sc.yv(-n, x_num))
    sign = (-1)**n
    print(f"    n={n}: Y_{{-{n}}}={Yn:.6f},  (-1)^n*Y_{n}={sign*Yp:.6f},  "
          f"match={abs(Yn - sign*Yp) < 1e-10}")

print("""
  WRONSKIAN (linear independence measure):
    W[J_nu, Y_nu](x) = J_nu * Y_nu' - J_nu' * Y_nu = 2/(pi*x)
    Non-zero for all x > 0 => J_nu and Y_nu are ALWAYS independent.
    (This is why Y_nu is the "right" second solution even though it diverges.)

  ASYMPTOTIC FORMS (x >> 1):
    J_nu(x) ~ sqrt(2/(pi*x)) * cos(x - nu*pi/2 - pi/4)
    Y_nu(x) ~ sqrt(2/(pi*x)) * sin(x - nu*pi/2 - pi/4)
    Both oscillate with DECAYING amplitude (1/sqrt(x)).
    Like damped cosine/sine -- phase offset by pi/2 between J and Y.
    |J_nu| and |Y_nu| both -> 0 as x -> inf.

  SMALL-x BEHAVIOR (x << 1):
    J_0(x) ~ 1 - x^2/4 + ...      (finite, = 1 at x=0)
    J_n(x) ~ (1/n!) * (x/2)^n      (-> 0 as x^n)
    Y_0(x) ~ (2/pi) * ln(x/2)      (-> -inf logarithmically)
    Y_n(x) ~ -(n-1)!/pi * (2/x)^n  (-> -inf as x^{-n} for n>=1)
""")

# Precise numerical values near x=0
print("  NUMERICAL NEAR x=0 (catastrophic cancellation warning):")
for x_val in [1.0, 0.1, 0.01, 1e-6]:
    J0 = sc.jv(0, x_val)
    Y0 = sc.yv(0, x_val)
    print(f"    x={x_val:.0e}: J_0={J0:.8f}, Y_0={Y0:.4f}")
print("  => Y_0 -> -inf as x->0: do NOT use Y_n if domain includes 0.")

print("""
  SPHERICAL BESSEL FUNCTIONS:
    j_n(x) = sqrt(pi/(2x)) * J_{n+1/2}(x)    [1st kind]
    y_n(x) = sqrt(pi/(2x)) * Y_{n+1/2}(x)    [2nd kind]
    From 3D Laplacian in spherical coordinates.
    Applications: QM hydrogen atom, acoustic scattering, Mie theory.

    j_0(x) = sin(x)/x                          [sinc function!]
    j_1(x) = sin(x)/x^2 - cos(x)/x
    y_0(x) = -cos(x)/x
    y_1(x) = -cos(x)/x^2 - sin(x)/x
    Wronskian: W[j_n, y_n] = 1/x^2
""")

x_sph = np.linspace(0.01, 15, 500)
j0_sph = np.sin(x_sph)/x_sph
j1_sph = np.sin(x_sph)/x_sph**2 - np.cos(x_sph)/x_sph
j0_sc  = sc.spherical_jn(0, x_sph)   # scipy verify

print("  Spherical j_0(x) = sin(x)/x vs scipy: max error =",
      f"{np.max(np.abs(j0_sph - j0_sc)):.2e}")

print("""
  MODIFIED BESSEL FUNCTIONS (imaginary argument):
    I_nu(x) = i^(-nu) * J_nu(ix) = SUM_{m=0}^{inf} 1/(m!*Gamma(m+nu+1)) * (x/2)^(2m+nu)
    K_nu(x) = (pi/2) * (I_{-nu} - I_nu) / sin(nu*pi)
    I_nu: exponentially GROWING,  K_nu: exponentially DECAYING.

    Small x: I_nu ~ (1/n!) * (x/2)^nu  (same as J_nu)
    Large x: I_nu ~ exp(x)/sqrt(2*pi*x)  (grows!)
             K_nu ~ sqrt(pi/(2x)) * exp(-x)  (decays)

    Application in fiber cladding (evanescent field):
    Cladding field ~ K_m(kappa*r) which decays away from core.
    kappa = sqrt(beta^2 - n_clad^2 * k0^2)  > 0 for guided mode.
""")

print("  Modified Bessel at x=2.0:")
x_mod = 2.0
for n in range(4):
    In = sc.iv(n, x_mod)
    Kn = sc.kv(n, x_mod)
    print(f"    n={n}: I_{n}={In:.6f},  K_{n}={Kn:.6f}")

# ============================================================
# S2: JAVA REFERENCE
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: JAVA REFERENCE")
print(SEP)

java_code = r"""
  // ---- JAVA BASICS ----

  // Primitive types (stack-allocated, not objects):
  byte   b = 127;              // 8-bit signed
  short  s = 32767;            // 16-bit signed
  int    i = 2147483647;       // 32-bit signed (default integer)
  long   l = 9223372036854775807L;  // 64-bit signed (suffix L)
  float  f = 3.14f;            // 32-bit IEEE 754 (suffix f)
  double d = 3.14159265358979; // 64-bit IEEE 754 (default float)
  char   c = 'A';              // 16-bit Unicode (not 8-bit like C)
  boolean flag = true;         // true or false (NOT 0/1)

  // Wrapper types (heap-allocated, used in Collections):
  Integer boxed = 42;          // auto-boxing: int -> Integer
  int unboxed = boxed;         // auto-unboxing: Integer -> int

  // String (immutable, reference type):
  String s1 = "hello";
  String s2 = "world";
  String s3 = s1 + " " + s2;      // "hello world" (creates new String)
  String s4 = String.format("pi=%.4f", Math.PI);
  boolean eq = s1.equals(s2);     // ALWAYS use .equals(), not ==
  // s1 == s2 compares REFERENCES (object identity), not content!
  int len = s1.length();
  char ch = s1.charAt(0);         // 'h'
  String upper = s1.toUpperCase();
  String[] parts = "a,b,c".split(",");  // ["a","b","c"]

  // ---- OOP: CLASS ----
  public class Animal {
      private String name;   // encapsulated field
      private int age;

      public Animal(String name, int age) {  // constructor
          this.name = name;
          this.age = age;
      }

      public String getName() { return name; }  // getter
      public void setName(String name) { this.name = name; }

      public String speak() { return "..."; }  // overrideable

      @Override
      public String toString() {
          return "Animal(" + name + ", age=" + age + ")";
      }
  }

  // ---- INHERITANCE ----
  public class Dog extends Animal {
      private String breed;

      public Dog(String name, int age, String breed) {
          super(name, age);    // call parent constructor
          this.breed = breed;
      }

      @Override
      public String speak() { return "Woof!"; }  // polymorphism
  }

  // ---- INTERFACE ----
  public interface Drawable {
      void draw();             // abstract (no body)
      default void clear() {  // default method (Java 8+)
          System.out.println("Clearing...");
      }
  }

  public class Circle implements Drawable {
      @Override
      public void draw() { System.out.println("Drawing circle"); }
  }

  // ---- ABSTRACT CLASS ----
  // Between interface (all abstract) and concrete class:
  public abstract class Shape {
      abstract double area();  // must implement in subclass
      public void print() {    // can have concrete methods
          System.out.println("Area: " + area());
      }
  }

  // ---- GENERICS ----
  public class Box<T> {
      private T content;
      public Box(T content) { this.content = content; }
      public T get() { return content; }
  }
  Box<Integer> intBox = new Box<>(42);
  Box<String>  strBox = new Box<>("hello");

  // ---- COLLECTIONS ----
  import java.util.*;
  List<String>    list = new ArrayList<>();  // dynamic array
  list.add("a"); list.add("b"); list.get(0); list.size();

  Map<String, Integer> map = new HashMap<>();
  map.put("key", 42);
  int val = map.getOrDefault("key", 0);
  map.containsKey("key");

  Set<Integer> set = new HashSet<>();  // unique elements, no order
  Set<Integer> tset = new TreeSet<>();  // sorted

  Deque<Integer> stack = new ArrayDeque<>();  // use as stack/queue
  stack.push(1); stack.pop(); stack.peek();

  // ---- STREAMS + LAMBDAS (Java 8+) ----
  List<Integer> nums = List.of(1, 2, 3, 4, 5);
  int sum = nums.stream()
                .filter(n -> n % 2 == 0)
                .mapToInt(Integer::intValue)
                .sum();  // = 6

  List<String> names = List.of("Alice", "Bob", "Charlie");
  names.stream()
       .filter(s -> s.length() > 3)
       .map(String::toUpperCase)
       .sorted()
       .forEach(System.out::println);  // ALICE, CHARLIE

  // ---- EXCEPTION HANDLING ----
  try {
      int result = 10 / 0;
  } catch (ArithmeticException e) {
      System.out.println("Math error: " + e.getMessage());
  } catch (Exception e) {
      System.out.println("Generic: " + e);
  } finally {
      System.out.println("Always runs");
  }

  // Checked vs unchecked:
  // Checked: must declare or catch (IOException, SQLException)
  // Unchecked: RuntimeException subclasses (NullPointerException, etc.)

  // ---- CONCURRENCY ----
  Thread t = new Thread(() -> System.out.println("Thread!"));
  t.start();
  t.join();  // wait for thread to finish

  // ExecutorService (preferred over raw Thread):
  ExecutorService exec = Executors.newFixedThreadPool(4);
  Future<Integer> future = exec.submit(() -> {
      Thread.sleep(100);
      return 42;
  });
  int result = future.get();  // blocks until done
  exec.shutdown();

  // synchronized keyword:
  synchronized(this) { /* critical section */ }
"""
print(java_code)

print("""  JAVA vs PYTHON vs C COMPARISON:
  Feature            Java                Python              C
  ----------------   ------------------  ------------------  ------------------
  Typing             Static, strong      Dynamic, strong     Static, weak
  Memory             GC (JVM)            GC (CPython RC)     Manual (malloc/free)
  OOP                Everything is class Optional, flexible  Struct-based
  Interfaces         interface keyword   ABC / duck typing   Function pointers
  Generics           <T> at compile time No (duck typed)     No (void*)
  Exceptions         Checked+unchecked   All unchecked       errno/return codes
  Concurrency        Threads+locks       GIL limits threads  pthreads
  Null safety        NullPointerException None dereference   Undefined behavior
  Build              javac + jar/maven   No compile needed   cc/gcc + make
  Speed              ~2-5x slower than C ~50-100x C          Fastest
  Use case           Enterprise/Android  ML/scripting/fast   Systems/embedded
""")

# ============================================================
# S3: JAVASCRIPT REFERENCE
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: JAVASCRIPT REFERENCE")
print(SEP)

js_code = r"""
  // ---- VARIABLES AND SCOPING ----
  var x = 1;     // function-scoped, HOISTED (moved to top of function)
  let y = 2;     // block-scoped, NOT hoisted usably (TDZ)
  const z = 3;   // block-scoped, cannot reassign (but object contents mutable)

  // HOISTING example:
  console.log(a);  // undefined (NOT ReferenceError) -- var is hoisted
  var a = 5;

  console.log(b);  // ReferenceError: Cannot access 'b' before init
  let b = 5;       // Temporal Dead Zone (TDZ)

  // ---- TYPES ----
  typeof 42           // "number"   (no int vs float: all IEEE 754 double)
  typeof "hi"         // "string"
  typeof true         // "boolean"
  typeof undefined    // "undefined"
  typeof null         // "object"   <-- famous bug, kept for compatibility
  typeof {}           // "object"
  typeof []           // "object"   (arrays are objects!)
  typeof function(){}  // "function"
  Array.isArray([])   // true  <-- correct way to check

  // Type coercion traps:
  "5" + 3    // "53"   (+ prefers string concatenation)
  "5" - 3    // 2      (- forces numeric)
  [] + {}    // "[object Object]"  (both convert to string)
  null == undefined   // true  (loose equality)
  null === undefined  // false (strict equality -- ALWAYS USE ===)

  // ---- FUNCTIONS ----
  function greet(name) { return "Hello, " + name; }  // declaration (hoisted)
  const greet2 = function(name) { return "Hi, " + name; };  // expression
  const greet3 = (name) => "Hey, " + name;  // arrow function (ES6)

  // Arrow functions DO NOT have their own 'this':
  const obj = {
      name: "obj",
      regular: function() { return this.name; },  // "obj"
      arrow:   () => this.name,                   // undefined or global
  };

  // ---- CLOSURES ----
  function makeCounter() {
      let count = 0;          // count is in closure scope
      return function() {
          count++;
          return count;
      };
  }
  const counter = makeCounter();
  counter();  // 1
  counter();  // 2  -- count persists in closure

  // ---- PROTOTYPE CHAIN ----
  function Animal(name) { this.name = name; }
  Animal.prototype.speak = function() { return this.name + " speaks"; };

  function Dog(name) { Animal.call(this, name); }
  Dog.prototype = Object.create(Animal.prototype);  // inherit
  Dog.prototype.constructor = Dog;

  // ES6 class syntax (syntactic sugar over prototypes):
  class Animal {
      constructor(name) { this.name = name; }
      speak() { return this.name + " speaks"; }
  }
  class Dog extends Animal {
      constructor(name) { super(name); }
      bark() { return "Woof!"; }
  }

  // ---- 'this' KEYWORD ----
  // 'this' = the object that called the function
  // In regular function: depends on call site
  // In arrow function:   lexically inherited (parent scope)
  // In class method:    the instance
  // In event handler:   the DOM element
  // setTimeout callback: global (window in browser, global in Node)
  // Fix: const self = this;  OR  use arrow functions  OR  .bind(this)

  // ---- DESTRUCTURING ----
  const [a, b, ...rest] = [1, 2, 3, 4, 5];  // a=1, b=2, rest=[3,4,5]
  const {name, age = 0} = {name: "Alice"};   // default values
  const {x: px, y: py} = point;             // renaming

  // ---- SPREAD AND REST ----
  const arr1 = [1, 2, 3];
  const arr2 = [...arr1, 4, 5];      // [1,2,3,4,5]  (spread)
  const merged = {...obj1, ...obj2}; // shallow merge

  function sum(...args) {            // rest parameter
      return args.reduce((a,b) => a+b, 0);
  }

  // ---- PROMISES AND ASYNC/AWAIT ----
  const p = new Promise((resolve, reject) => {
      setTimeout(() => resolve("done"), 1000);
  });
  p.then(val => console.log(val))  // "done" after 1s
   .catch(err => console.error(err));

  // Promise.all: wait for multiple
  const [a, b] = await Promise.all([fetchA(), fetchB()]);

  async function fetchData(url) {
      try {
          const response = await fetch(url);
          const data     = await response.json();
          return data;
      } catch (error) {
          console.error("Fetch failed:", error);
      }
  }

  // ---- EVENT LOOP ----
  // Call stack: sync code
  // Microtask queue: Promise .then(), queueMicrotask()  (PRIORITY)
  // Macrotask queue: setTimeout, setInterval, I/O callbacks
  // Order: call stack -> ALL microtasks -> one macrotask -> repeat

  setTimeout(() => console.log("macro"), 0);
  Promise.resolve().then(() => console.log("micro"));
  console.log("sync");
  // Output: "sync", "micro", "macro"

  // ---- NODE.JS MODULES ----
  // CommonJS (older):
  const fs = require("fs");
  module.exports = { myFunc };

  // ES Modules (modern, .mjs or "type":"module" in package.json):
  import { readFile } from "fs/promises";
  export function myFunc() {}
  export default class MyClass {}
"""
print(js_code)

print("""  JS vs PYTHON comparison:
  Feature            JavaScript               Python
  ----------------   ----------------------   -------------------
  Typing             Dynamic, weak coercion   Dynamic, strong
  Null               null AND undefined       None only
  Equality           == (coerce) vs === (strict) == (always value)
  This               Context-dependent        self (explicit)
  Async              Promise / async-await    asyncio / async-await
  Classes            Prototype-based sugar    True classes
  Modules            CJS or ESM               import / pip
  Numbers            1 type (IEEE 754 double) int + float
  Truthy             0,"",[],{} = truthy!     0,"",[],{} = falsy
  Arrays             Object, sparse ok        list, numpy array
  Runtime            Browser + Node.js        CPython, PyPy
""")

# ============================================================
# S4: VERILOG HDL
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: VERILOG HDL")
print(SEP)

verilog_code = r"""
  // ---- MODULE BASICS ----
  module my_module (
      input  wire       clk,      // clock
      input  wire       rst_n,    // active-low reset
      input  wire [7:0] data_in,  // 8-bit input bus [MSB:LSB]
      output reg  [7:0] data_out  // registered output
  );

  // ---- WIRE vs REG ----
  // wire: combinational connection (driven by continuous assign or module output)
  // reg:  can hold value in always block (does NOT mean flip-flop in synthesis)
  wire [7:0] intermediate;
  assign intermediate = data_in + 8'd1;  // continuous assignment

  // ---- ALWAYS BLOCK: SEQUENTIAL (flip-flops) ----
  always @(posedge clk or negedge rst_n) begin
      if (!rst_n)          // async reset
          data_out <= 8'b0;
      else
          data_out <= intermediate;  // non-blocking: sample RHS at clock edge
  end

  // ---- BLOCKING (=) vs NON-BLOCKING (<=) ----
  // NON-BLOCKING (<=): use in sequential (clocked) always blocks
  //   RHS evaluated NOW, LHS updated END of time step
  //   All NBAs happen simultaneously -> models flip-flop behavior
  // BLOCKING (=): use in combinational always blocks or testbenches
  //   RHS evaluated and LHS updated immediately -> sequential execution

  // WRONG (race condition):
  always @(posedge clk) begin
      a = b;   // blocking in seq -- synthesizes, but dangerous
      b = a;   // a already updated above!
  end

  // CORRECT (swap with non-blocking):
  always @(posedge clk) begin
      a <= b;  // both RHS sampled at same time
      b <= a;  // then both updated: true swap
  end

  // ---- ALWAYS BLOCK: COMBINATIONAL ----
  reg [7:0] mux_out;
  always @(*) begin       // @(*) = sensitivity to all inputs
      case (sel)
          2'b00: mux_out = a;
          2'b01: mux_out = b;
          2'b10: mux_out = c;
          default: mux_out = 8'hFF;
      endcase
  end

  // ---- 4-BIT ALU ----
  module alu_4bit (
      input  [3:0] a,
      input  [3:0] b,
      input  [2:0] op,       // operation select
      output reg [3:0] result,
      output reg       carry,
      output reg       zero
  );
      always @(*) begin
          carry = 0;
          case (op)
              3'b000: {carry, result} = a + b;   // ADD
              3'b001: result = a - b;              // SUB
              3'b010: result = a & b;              // AND
              3'b011: result = a | b;              // OR
              3'b100: result = a ^ b;              // XOR
              3'b101: result = ~a;                 // NOT a
              3'b110: result = a << 1;             // shift left
              3'b111: result = a >> 1;             // shift right
              default: result = 4'b0;
          endcase
          zero = (result == 4'b0) ? 1'b1 : 1'b0;
      end
  endmodule

  // ---- PARAMETERS (generics in Verilog) ----
  module shift_reg #(
      parameter WIDTH = 8,
      parameter DEPTH = 4
  ) (
      input              clk,
      input  [WIDTH-1:0] d,
      output [WIDTH-1:0] q
  );
      reg [WIDTH-1:0] pipeline [DEPTH-1:0];
      integer k;
      always @(posedge clk) begin
          pipeline[0] <= d;
          for (k=1; k<DEPTH; k=k+1)
              pipeline[k] <= pipeline[k-1];
      end
      assign q = pipeline[DEPTH-1];
  endmodule

  // ---- TESTBENCH ----
  module tb_alu;
      reg  [3:0] a, b;
      reg  [2:0] op;
      wire [3:0] result;
      wire       carry, zero;

      alu_4bit dut (.a(a),.b(b),.op(op),.result(result),
                    .carry(carry),.zero(zero));

      initial begin
          $dumpfile("alu.vcd");    // waveform output
          $dumpvars(0, tb_alu);

          a=4'd5; b=4'd3; op=3'b000; #10;  // ADD: 5+3=8
          $display("5+3=%0d carry=%b", result, carry);

          a=4'd2; b=4'd9; op=3'b000; #10;  // ADD with carry: 2+9=11
          $display("2+9=%0d carry=%b", result, carry);

          a=4'd7; b=4'd7; op=3'b100; #10;  // XOR: 7^7=0
          $display("7^7=%0d zero=%b", result, zero);

          $finish;
      end
  endmodule
"""
print(verilog_code)

print("""  VERILOG SYNTHESIS RULES:
  Construct          Synthesizes to       Notes
  ----------------   ------------------   ----------------------------
  assign wire = ...  Combinational logic  Always-on (continuous)
  always @(posedge)  Flip-flops (DFF)     Non-blocking (<=) required
  always @(*)        Combinational gate   Blocking (=) ok; list all inputs
  if/case in always  Multiplexers         Missing cases -> latch (bad!)
  for loop (const)   Unrolled gates       Loop bounds must be constant
  integer (runtime)  NOT synthesizable    Use only in testbenches
  $display, $finish  NOT synthesizable    Simulation only
  initial block      NOT synthesizable*   Sim only (* FPGA has exceptions)

  LATCH vs FLIP-FLOP:
  Latch: level-sensitive, transparent when enabled  (usually unintentional)
  DFF:   edge-triggered, samples only at clock edge  (intentional)
  To avoid latches: cover ALL cases in if/case, or use else/default.

  VERILOG vs VHDL:
  Verilog: C-like syntax, less verbose, industry standard in the US
  VHDL:    Ada-like, strongly typed, preferred in Europe/aerospace
  SystemVerilog: Verilog++ with classes, interfaces, assertions (modern)
""")

# ============================================================
# MATPLOTLIB -- 4-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#F8F8F0")
gs0 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                        top=0.93, bottom=0.05, left=0.06, right=0.97)

ax_jn    = fig.add_subplot(gs0[0, 0])
ax_yn    = fig.add_subplot(gs0[0, 1])
ax_sph   = fig.add_subplot(gs0[0, 2])
ax_mod   = fig.add_subplot(gs0[1, 0])
ax_asym  = fig.add_subplot(gs0[1, 1])
ax_wron  = fig.add_subplot(gs0[1, 2])

fig.suptitle("Bessel Functions: Precise Symmetry, Singular Behavior, Asymptotics",
             fontsize=13, fontweight="bold", color="#1a1a2e")

xb = np.linspace(0.001, 20, 2000)
xb_pos = np.linspace(0.1, 20, 2000)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ---- AX_JN: J_n and J_{-n} ----
ax = ax_jn
ax.set_facecolor("#F0F4FF")
for n, col in [(0,"#1f77b4"),(1,"#ff7f0e"),(2,"#2ca02c")]:
    Jp = sc.jv(n, xb)
    Jn = sc.jv(-n, xb)
    ax.plot(xb, Jp, color=col, lw=1.8, label=f"J_{n}(x)")
    ax.plot(xb, Jn, color=col, lw=1.0, ls="--", alpha=0.6,
            label=f"J_{{-{n}}}=(-1)^{n}*J_{n}")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(-0.5, 1.05)
ax.set_title(r"$J_{-n}(x) = (-1)^n J_n(x)$ -- symmetry", fontsize=10)
ax.legend(fontsize=6.5, loc="upper right", ncol=2)
ax.grid(alpha=0.2)
ax.set_xlabel("x"); ax.set_ylabel("Amplitude")

# ---- AX_YN: Y_n singularity ----
ax = ax_yn
ax.set_facecolor("#FFF0F0")
for n, col in [(0,"#1f77b4"),(1,"#ff7f0e"),(2,"#2ca02c")]:
    Yn = sc.yv(n, xb_pos)
    ax.plot(xb_pos, np.clip(Yn, -5, 5), color=col, lw=1.5, label=f"Y_{n}(x)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, 15)
ax.set_ylim(-3, 1)
ax.set_title(r"$Y_n(x)$: Singular at $x=0$ (Neumann)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlabel("x")
ax.text(0.02, 0.97, "Y_0 ~ (2/pi)ln(x) as x->0\nY_n ~ -(n-1)!/pi*(2/x)^n",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_SPH: spherical Bessel ----
ax = ax_sph
ax.set_facecolor("#F0FFF0")
xsp = np.linspace(0.01, 20, 1000)
for n, col in [(0,"#1f77b4"),(1,"#ff7f0e"),(2,"#2ca02c"),(3,"#d62728")]:
    jn = sc.spherical_jn(n, xsp)
    ax.plot(xsp, jn, color=col, lw=1.5, label=f"j_{n}(x)")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(-0.4, 1.05)
ax.set_title(r"Spherical Bessel $j_n(x) = \sqrt{\pi/2x}\,J_{n+1/2}(x)$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlabel("x")
ax.text(0.02, 0.04, r"$j_0(x) = \sin(x)/x$  (sinc!)",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="#ffffcc", ec="#bbb", pad=2))

# ---- AX_MOD: modified Bessel I_n and K_n ----
ax = ax_mod
ax.set_facecolor("#FFF5FF")
xm = np.linspace(0.01, 5, 500)
for n, col in [(0,"#1f77b4"),(1,"#ff7f0e"),(2,"#2ca02c")]:
    In = sc.iv(n, xm)
    Kn = sc.kv(n, xm)
    ax.semilogy(xm, In, color=col, lw=1.5, label=f"I_{n}(x)")
    ax.semilogy(xm, Kn, color=col, lw=1.5, ls="--", label=f"K_{n}(x)")
ax.set_ylim(1e-3, 1e3)
ax.set_title("Modified Bessel: $I_n$ (grow), $K_n$ (decay)", fontsize=10)
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.2, which="both")
ax.set_xlabel("x")
ax.text(0.55, 0.95, "Fiber cladding:\nfield ~ K_m(kappa*r)",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="white", ec="#1f77b4", pad=2))

# ---- AX_ASYM: asymptotic comparison ----
ax = ax_asym
ax.set_facecolor("#FFFFF0")
xa = np.linspace(1, 30, 1000)
J0_exact = sc.jv(0, xa)
J0_asym  = np.sqrt(2/(np.pi*xa)) * np.cos(xa - np.pi/4)
ax.plot(xa, J0_exact, "#1f77b4", lw=1.8, label="J_0(x) exact")
ax.plot(xa, J0_asym,  "#d62728", lw=1.2, ls="--", label=r"$\sqrt{2/\pi x}\cos(x-\pi/4)$")
err = np.abs(J0_exact - J0_asym)
ax2 = ax.twinx()
ax2.semilogy(xa, err, "#2ca02c", lw=0.8, alpha=0.6, label="Error")
ax2.set_ylabel("Asymptotic error", fontsize=8, color="#2ca02c")
ax2.tick_params(axis="y", colors="#2ca02c")
ax.set_title(r"Asymptotic: $J_0\approx\sqrt{2/\pi x}\cos(x-\pi/4)$", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.2)
ax.set_xlabel("x")

# ---- AX_WRON: Wronskian = 2/pi*x ----
ax = ax_wron
ax.set_facecolor("#F0F8FF")
xw = np.linspace(0.5, 15, 500)
J0w = sc.jv(0, xw)
J1w = sc.jv(1, xw)
Y0w = sc.yv(0, xw)
Y1w = sc.yv(1, xw)
# W[J0, Y0] = J0*Y0' - J0'*Y0 = J0*(-Y1) - (-J1)*Y0
W_numerical = J0w*(-Y1w) - (-J1w)*Y0w
W_exact     = 2/(np.pi*xw)
ax.plot(xw, W_numerical, "#1f77b4", lw=1.8, label="W[J_0,Y_0] numerical")
ax.plot(xw, W_exact,     "#d62728", lw=1.2, ls="--", label=r"$2/\pi x$ (exact)")
ax.set_ylim(0, 1.5)
ax.set_title(r"Wronskian $W[J_0,Y_0] = 2/\pi x$ (independence)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlabel("x")
ax.text(0.55, 0.9, "W != 0 for all x>0\n=> J_0, Y_0 independent",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
