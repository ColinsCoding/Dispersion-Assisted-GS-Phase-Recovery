"""
start_tunnel.py  —  Jalali Lab Optical Dashboard  +  Cloudflare Tunnel
=======================================================================
What this does (in order):
  1. Checks cloudflared is installed; prints install command if not.
  2. Starts the Flask app in the background.
  3. Starts a Cloudflare Tunnel, captures the public HTTPS URL.
  4. Patches the live-dashboard badge in README.md with that URL.
  5. Commits and pushes the README update to GitHub.
  6. Keeps running (Ctrl-C stops both Flask and the tunnel).

Usage:
  python start_tunnel.py

Requirements:
  cloudflared  —  winget install --id Cloudflare.cloudflared
  git remote configured with push access (already done)
"""

import subprocess, re, sys, time, signal, pathlib, shutil, threading, os

REPO   = pathlib.Path(__file__).parent
README = REPO / "README.md"
APP    = REPO / "optical_dashboard" / "app.py"
PYTHON = sys.executable

# ── Badge regex patterns ──────────────────────────────────────────────────────
# Matches the URL inside the "Live Dashboard" badge link
_BADGE_URL = re.compile(
    r'(!\[Live Dashboard\][^\]]*\]\()https://[^\)]+(\))',
    re.MULTILINE,
)
_HEALTH_URL = re.compile(
    r'(!\[Health\][^\]]*\]\()https://[^\)]+(/health\))',
    re.MULTILINE,
)
# Cloudflare prints the URL to stderr like:
#   2024/... INF +----------------------------+
#   2024/... INF |  https://xxxx.trycloudflare.com  |
_CF_URL = re.compile(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com')


def check_cloudflared():
    if shutil.which("cloudflared"):
        ver = subprocess.run(["cloudflared", "--version"],
                             capture_output=True, text=True).stdout.strip()
        print(f"[ok] cloudflared found: {ver}")
        return True
    print("\n[!] cloudflared is NOT installed.")
    print("    Install it with ONE of these commands:\n")
    print("    winget install --id Cloudflare.cloudflared        (Windows 10/11)")
    print("    OR download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/\n")
    return False


def start_flask():
    proc = subprocess.Popen(
        [PYTHON, str(APP)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(REPO / "optical_dashboard"),
    )
    # Forward Flask output on a background thread
    def _pipe():
        for line in proc.stdout:
            sys.stdout.write("[flask] " + line.decode(errors="replace"))
            sys.stdout.flush()
    threading.Thread(target=_pipe, daemon=True).start()
    # Wait until /health responds
    import urllib.request
    for _ in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=2)
            print("[ok] Flask is up at http://127.0.0.1:5000")
            return proc
        except Exception:
            pass
    print("[!] Flask did not start in 30 s — check errors above.")
    proc.terminate()
    sys.exit(1)


def start_tunnel():
    """Start cloudflared, return (process, public_url)."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    url = None
    print("[..] Waiting for Cloudflare tunnel URL (up to 30 s)...")
    start = time.time()
    for line in proc.stdout:
        sys.stdout.write("[cf] " + line)
        sys.stdout.flush()
        m = _CF_URL.search(line)
        if m:
            url = m.group(0)
            break
        if time.time() - start > 30:
            break
    if not url:
        print("[!] Could not capture Cloudflare URL.")
        proc.terminate()
        sys.exit(1)
    print(f"\n[ok] Public URL: {url}\n")
    return proc, url


def patch_readme(url: str):
    text = README.read_text(encoding="utf-8")
    # Replace live-dashboard badge link
    new_text = _BADGE_URL.sub(rf"\g<1>{url}\g<2>", text)
    # Replace health badge link
    new_text = _HEALTH_URL.sub(rf"\g<1>{url}\g<2>", new_text)
    if new_text == text:
        print("[..] README badge URL unchanged (already correct).")
        return
    README.write_text(new_text, encoding="utf-8")
    print(f"[ok] README.md badge updated -> {url}")


def git_push(url: str):
    os.chdir(str(REPO))
    subprocess.run(["git", "add", "README.md"], check=True)
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True
    )
    if result.returncode == 0:
        print("[..] No README change to commit (URL same as before).")
        return
    subprocess.run(
        ["git", "commit", "-m", f"chore: live tunnel URL -> {url}"],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"[ok] README pushed. GitHub badge now points to:\n     {url}")


def main():
    print("=" * 60)
    print("  Jalali Lab Dashboard  +  Cloudflare Tunnel")
    print("=" * 60)

    if not check_cloudflared():
        sys.exit(1)

    flask_proc  = start_flask()
    cf_proc, url = start_tunnel()

    patch_readme(url)
    git_push(url)

    print("\n" + "=" * 60)
    print(f"  LIVE:  {url}")
    print(f"  LOCAL: http://localhost:5000")
    print(f"  Ctrl-C to stop both Flask and the tunnel")
    print("=" * 60 + "\n")

    # Wait until Ctrl-C
    def _shutdown(sig, frame):
        print("\n[..] Shutting down...")
        cf_proc.terminate()
        flask_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep reading cloudflared output so it doesn't block
    for line in cf_proc.stdout:
        sys.stdout.write("[cf] " + line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
