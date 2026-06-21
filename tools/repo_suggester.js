#!/usr/bin/env node
/* repo_suggester.js -- scan the repo, pipe findings into C, write SUGGESTIONS.md.
 *
 * The "dream algorithm" you wake up to: a heuristic scanner (this file) that
 * finds concrete, actionable gaps in the codebase, hands them to a C ranking
 * engine (tools/score.c) over a pipe, and writes the ranked result to
 * SUGGESTIONS.md so you can read the top items in the morning and act on them.
 *
 * No LLM, no network -- pure static heuristics, so it is honest and repeatable.
 * Run:  node tools/repo_suggester.js     (auto-compiles tools/score.c if needed)
 */
"use strict";
const fs = require("fs");
const path = require("path");
const { execFileSync, spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const SCORE_SRC = path.join(__dirname, "score.c");
const SCORE_EXE = path.join(__dirname, process.platform === "win32" ? "score.exe" : "score");

// ---------- tiny fs helpers ----------
const read = (f) => fs.readFileSync(f, "utf8");
const listFiles = (dir, ext) => {
  const d = path.join(ROOT, dir);
  if (!fs.existsSync(d)) return [];
  return fs.readdirSync(d).filter((f) => f.endsWith(ext)).map((f) => path.join(dir, f));
};

// ---------- the heuristics: each pushes {severity, effort, text} ----------
const candidates = [];
const add = (severity, effort, text) => candidates.push({ severity, effort, text });

const pyModules = listFiles("dgs", ".py").filter((f) => !f.endsWith("__init__.py"));
const griffMods = listFiles("griffiths", ".py").filter((f) => !f.endsWith("__init__.py"));
const testFiles = listFiles("tests", ".py");
const testText = testFiles.map((f) => read(path.join(ROOT, f))).join("\n");

// 1. package modules with no test that references them
for (const mod of [...pyModules, ...griffMods]) {
  const name = path.basename(mod, ".py");
  const hasNamedTest = fs.existsSync(path.join(ROOT, "tests", `test_${name}.py`));
  const mentioned = new RegExp(`\\b${name}\\b`).test(testText);
  if (!hasNamedTest && !mentioned)
    add(7, 3, `No test covers ${mod} -- add tests/test_${name}.py with an analytic check`);
}

// 2. public functions missing a docstring (heuristic: def not followed by a triple-quote)
for (const mod of [...pyModules, ...griffMods]) {
  const lines = read(path.join(ROOT, mod)).split(/\r?\n/);
  const missing = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^def ([a-zA-Z]\w*)\s*\(/); // public, module-level
    if (!m) continue;
    let j = i; // walk to the end of the (possibly multi-line) signature
    while (j < lines.length && !/:\s*(#.*)?$/.test(lines[j])) j++;
    let k = j + 1;
    while (k < lines.length && lines[k].trim() === "") k++;
    if (k < lines.length && !/^\s*(["']{3})/.test(lines[k])) missing.push(m[1]);
  }
  if (missing.length)
    add(Math.min(8, 2 + missing.length), 2,
      `${mod}: ${missing.length} public fn(s) lack a docstring (${missing.slice(0, 3).join(", ")}${missing.length > 3 ? ", ..." : ""})`);
}

// 3. TODO / FIXME / HACK markers in source
for (const mod of [...pyModules, ...griffMods, ...listFiles("tools", ".js")]) {
  const lines = read(path.join(ROOT, mod)).split(/\r?\n/);
  lines.forEach((ln, i) => {
    if (/\b(TODO|FIXME|XXX|HACK)\b/.test(ln))
      add(5, 1, `${mod}:${i + 1} unresolved marker -> ${ln.trim().slice(0, 70)}`);
  });
}

// 4. stray .py files at the repo root (policy: root is config-only)
for (const f of fs.readdirSync(ROOT)) {
  if (f.endsWith(".py") && f !== "setup.py" && fs.statSync(path.join(ROOT, f)).isFile())
    add(6, 2, `Root-level ${f}: move into dgs/ or delete (root should be config-only)`);
}

// 5. oversized modules worth splitting
for (const mod of [...pyModules, ...griffMods]) {
  const n = read(path.join(ROOT, mod)).split(/\r?\n/).length;
  if (n > 400) add(4, 5, `${mod} is ${n} lines -- consider splitting into focused modules`);
}

if (candidates.length === 0) {
  fs.writeFileSync(path.join(ROOT, "SUGGESTIONS.md"),
    `# Repo suggestions\n\nGenerated ${new Date().toISOString()}\n\nNothing flagged -- the package is clean. 🎉\n`);
  console.log("no suggestions; wrote a clean bill of health");
  process.exit(0);
}

// ---------- compile the C ranker if needed ----------
function findCompiler() {
  // try cc/gcc on PATH, then known mingw64 location; return {cmd, env} or throw
  for (const cmd of ["cc", "gcc"]) {
    if (spawnSync(cmd, ["--version"]).status === 0) return { cmd, env: process.env };
  }
  const mingw = "C:\\msys64\\mingw64\\bin";
  if (fs.existsSync(path.join(mingw, "cc.exe")))
    return { cmd: path.join(mingw, "cc.exe"), env: { ...process.env, Path: mingw + ";" + process.env.Path } };
  throw new Error("no C compiler found (install gcc/clang, or msys2 mingw64)");
}
if (!fs.existsSync(SCORE_EXE) || fs.statSync(SCORE_SRC).mtimeMs > fs.statSync(SCORE_EXE).mtimeMs) {
  console.log("compiling tools/score.c ...");
  const { cmd, env } = findCompiler();
  execFileSync(cmd, ["-O2", "-o", SCORE_EXE, SCORE_SRC], { stdio: "inherit", env });
}

// ---------- pipe candidates -> C -> ranked output ----------
const payload = candidates.map((c) => `${c.severity} ${c.effort} ${c.text}`).join("\n") + "\n";
const ranked = spawnSync(SCORE_EXE, [], { input: payload, encoding: "utf8" });
if (ranked.status !== 0) { console.error("score engine failed:", ranked.stderr); process.exit(1); }

const rows = ranked.stdout.trim().split(/\r?\n/).map((ln) => {
  const [score, ...rest] = ln.split("\t");
  return { score: parseFloat(score), text: rest.join("\t") };
});

// ---------- write SUGGESTIONS.md ----------
let md = `# Repo suggestions (overnight scan)\n\n`;
md += `Generated ${new Date().toISOString()} -- ${rows.length} items, ranked by priority `;
md += `(impact^2 / effort, computed in C).\n\n`;
md += `Pipeline: \`repo_suggester.js\` (scan) -> \`score.c\` (rank) -> this file.\n\n`;
md += `| # | priority | suggestion |\n|---|---|---|\n`;
rows.forEach((r, i) => {
  md += `| ${i + 1} | ${r.score.toFixed(1)} | ${r.text.replace(/\|/g, "\\|")} |\n`;
});
md += `\n---\n_To act on these, paste the top few into the chat and we'll knock them out._\n`;

fs.writeFileSync(path.join(ROOT, "SUGGESTIONS.md"), md);
console.log(`wrote SUGGESTIONS.md with ${rows.length} ranked suggestions`);
console.log("top 3:");
rows.slice(0, 3).forEach((r, i) => console.log(`  ${i + 1}. [${r.score.toFixed(1)}] ${r.text}`));
