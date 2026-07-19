from pathlib import Path
p = Path("run_dryad_plots.py")
s = p.read_text(encoding="utf-8")
old = "plt.style.use(\"seaborn-darkgrid\")"
if old in s:
    new_block = (
        "try:\\n"
        "    import seaborn as sns\\n"
        "    sns.set_style(\"darkgrid\")\\n"
        "except Exception:\\n"
        "    plt.style.use(\"default\")\\n"
    )
    s = s.replace(old, new_block)
    p.write_text(s, encoding="utf-8")
    print('Patched run_dryad_plots.py: replaced seaborn style line with try/except block')
else:
    print('No exact seaborn style line found; no change made')
