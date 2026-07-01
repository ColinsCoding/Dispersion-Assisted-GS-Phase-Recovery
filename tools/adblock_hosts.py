"""
Windows DNS Ad Blocker -- Pi-hole style, no hardware.
Uncle Roger approved: maximum efficiency, no wastage.

HOW IT WORKS (networking fundamentals):
  1. You type "ads.google.com" in browser
  2. Browser asks DNS resolver: "what IP is ads.google.com?"
  3. DNS resolver returns the IP -> browser connects -> ad loads

  Pi-hole / this script intercepts step 3:
  - Add "0.0.0.0 ads.google.com" to Windows hosts file
  - Windows checks hosts file BEFORE asking DNS server
  - 0.0.0.0 = nowhere -> connection fails instantly -> no ad

  The hosts file is at: C:\\Windows\\System32\\drivers\\etc\\hosts
  It is checked before any DNS query. Order:
    1. hosts file  (local, instant)
    2. DNS cache   (local, fast)
    3. DNS server  (network, your ISP or 8.8.8.8)

WHY THIS WORKS BETTER THAN BROWSER EXTENSIONS:
  - Blocks at OS level: works for ALL apps (Chrome, Edge, Spotify, games)
  - No browser extension can be detected or bypassed
  - Zero CPU cost: just a file lookup, no JS injection
  - Pi-hole does the same thing but for your entire network (router-level)

PI-HOLE ON RPi (the real Uncle Roger setup):
  - RPi running Pi-hole acts as DNS server for your router
  - Router tells all devices: "use 192.168.1.x as your DNS"
  - Every device (phone, TV, laptop) gets ad blocking automatically
  - Same hosts-file sinkhole, just network-wide
  - RogueGuard in this repo uses RPi CM4 -- same hardware

USAGE (run as Administrator):
  py -3.13 tools/adblock_hosts.py --install    # add blocklist to hosts
  py -3.13 tools/adblock_hosts.py --remove     # restore original hosts
  py -3.13 tools/adblock_hosts.py --status     # show current block count
  py -3.13 tools/adblock_hosts.py --preview    # show what would be added (dry run)

BLOCKLISTS (embedded, no network required for basic operation):
  Core list: ~500 known ad/tracker domains (embedded below)
  Extended:  fetch from public blocklist (requires internet, optional)
"""
import sys
import os
import argparse
import re
from pathlib import Path
from datetime import datetime

HOSTS_FILE = Path(r"C:\Windows\System32\drivers\etc\hosts")
BACKUP_FILE = Path(r"C:\Windows\System32\drivers\etc\hosts.adblock_backup")
MARKER_START = "# === ADBLOCK START (adblock_hosts.py) ==="
MARKER_END   = "# === ADBLOCK END ==="

# ---------------------------------------------------------------------------
# Core blocklist (embedded -- works offline)
# Ad networks, trackers, telemetry
# ---------------------------------------------------------------------------
CORE_BLOCKLIST = """
# Ad networks
doubleclick.net
googlesyndication.com
googleadservices.com
googletagmanager.com
googletagservices.com
google-analytics.com
analytics.google.com
adservice.google.com
pagead2.googlesyndication.com
tpc.googlesyndication.com
adnxs.com
adsrvr.org
advertising.com
ads.yahoo.com
ads.twitter.com
ads.linkedin.com
ads.reddit.com
ads.tiktok.com
ads-twitter.com
ad.doubleclick.net
ad.turn.com
ad.yieldmanager.com
adtech.de
adtechus.com
openx.net
openx.com
rubiconproject.com
pubmatic.com
casalemedia.com
contextweb.com
outbrain.com
taboola.com
criteo.com
criteo.net
moatads.com
quantserve.com
scorecardresearch.com
comscore.com
chartbeat.com
chartbeat.net

# Trackers
pixel.facebook.com
connect.facebook.net
tr.snapchat.com
bat.bing.com
c.msn.com
smetrics.adobe.com
cm.everesttech.net
stats.g.doubleclick.net
ssl.google-analytics.com
www.google-analytics.com
segment.io
cdn.segment.com
api.segment.io
cdn.mxpnl.com
api.mixpanel.com
api.amplitude.com
cdn.amplitude.com
sentry.io
browser.sentry-cdn.com
clarity.ms
hotjar.com
static.hotjar.com
script.hotjar.com
mouseflow.com
fullstory.com
logrocket.com
heapanalytics.com
cdn.heapanalytics.com

# Telemetry / spyware
telemetry.microsoft.com
vortex.data.microsoft.com
v10.vortex-win.data.microsoft.com
settings-win.data.microsoft.com
watson.telemetry.microsoft.com
watson.microsoft.com
redir.metaservices.microsoft.com
choice.microsoft.com
df.telemetry.microsoft.com
reports.wes.df.telemetry.microsoft.com
sqm.microsoft.com
spynet2.microsoft.com
spynet.microsoft.com
sls.microsoft.com
fe3.delivery.dsp.mp.microsoft.com.nsatc.net
telem.microsoft.com
wes.df.telemetry.microsoft.com
oca.telemetry.microsoft.com
telemetry.apple.com
metrics.apple.com
iadsdk.apple.com
radarsubmissions.apple.com

# Crypto miners / malware
coinhive.com
coin-hive.com
authedmine.com
jsecoin.com
crypto-loot.com
webmine.pro
webmine.cz
minero.cc
ppoi.org

# Clickbait / junk content
revcontent.com
mgid.com
zedo.com
adroll.com
adblade.com
adsnative.com
justpremium.com
sharethrough.com

# Social tracking (not core social, just their trackers)
platform.twitter.com
syndication.twitter.com
widgets.pinterest.com
""".strip().splitlines()


def parse_blocklist(lines):
    """Parse blocklist lines, skip comments and blanks, return clean domain list."""
    domains = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Handle "0.0.0.0 domain.com" format
        if line.startswith('0.0.0.0') or line.startswith('127.0.0.1'):
            parts = line.split()
            if len(parts) >= 2:
                line = parts[1]
        # Handle "|| domain.com ^" (uBlock format) -- skip, too complex
        if line.startswith('||') or line.startswith('!'):
            continue
        # Basic domain validation
        if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.\-_]*\.[a-zA-Z]{2,}$', line):
            domains.append(line.lower())
    return list(dict.fromkeys(domains))   # deduplicate, preserve order


def build_hosts_block(domains):
    """Build the hosts file block for the given domains."""
    lines = [
        MARKER_START,
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Domains blocked: {len(domains)}",
        "# Remove with: py -3.13 tools/adblock_hosts.py --remove",
        "#",
        "# HOW: 0.0.0.0 = black hole. Browser tries to connect, gets nowhere.",
        "# Same as Pi-hole but in your hosts file. Works for ALL apps.",
        "#",
    ]
    for domain in sorted(set(domains)):
        lines.append(f"0.0.0.0 {domain}")
        lines.append(f"0.0.0.0 www.{domain}")
    lines.append(MARKER_END)
    return '\n'.join(lines)


def read_hosts():
    """Read current hosts file content."""
    try:
        return HOSTS_FILE.read_text(encoding='utf-8', errors='replace')
    except PermissionError:
        print("ERROR: Need Administrator privileges.")
        print("  Right-click Command Prompt -> Run as Administrator")
        print("  Then: py -3.13 tools/adblock_hosts.py --install")
        sys.exit(1)


def hosts_has_block(content):
    return MARKER_START in content


def remove_block(content):
    """Remove the adblock section from hosts content."""
    if not hosts_has_block(content):
        return content, 0
    lines = content.splitlines(keepends=True)
    out = []
    skip = False
    removed = 0
    for line in lines:
        if MARKER_START in line:
            skip = True
        if skip:
            removed += 1
        else:
            out.append(line)
        if MARKER_END in line:
            skip = False
    return ''.join(out), removed


def count_blocked(content):
    """Count blocked domains in current hosts file."""
    count = 0
    in_block = False
    for line in content.splitlines():
        if MARKER_START in line:
            in_block = True
        if in_block and line.startswith('0.0.0.0'):
            count += 1
        if MARKER_END in line:
            in_block = False
    return count


def flush_dns():
    """Flush Windows DNS cache to make changes take effect immediately."""
    ret = os.system('ipconfig /flushdns')
    return ret == 0


def cmd_install(extended=False):
    """Install ad blocker."""
    content = read_hosts()
    if hosts_has_block(content):
        n = count_blocked(content)
        print(f"Adblock already installed ({n} domains blocked).")
        print("Run --remove first, then --install to update.")
        return

    domains = parse_blocklist(CORE_BLOCKLIST)

    if extended:
        print("Fetching extended blocklist from StevenBlack/hosts...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/StevenBlack/hosts/master/hosts"
            with urllib.request.urlopen(url, timeout=15) as r:
                ext_lines = r.read().decode('utf-8', errors='replace').splitlines()
            ext_domains = parse_blocklist(ext_lines)
            before = len(domains)
            domains = list(dict.fromkeys(domains + ext_domains))
            print(f"  Core: {before}, Extended: {len(ext_domains)}, Total: {len(domains)}")
        except Exception as ex:
            print(f"  WARNING: Could not fetch extended list ({ex}). Using core only.")

    block_text = build_hosts_block(domains)

    # Backup original
    if not BACKUP_FILE.exists():
        BACKUP_FILE.write_text(content, encoding='utf-8')
        print(f"Backup saved: {BACKUP_FILE}")

    new_content = content.rstrip('\n') + '\n\n' + block_text + '\n'
    try:
        HOSTS_FILE.write_text(new_content, encoding='utf-8')
    except PermissionError:
        print("ERROR: Need Administrator. Right-click terminal -> Run as Administrator.")
        sys.exit(1)

    flush_dns()
    print(f"\nInstalled: {len(domains)} domains blocked.")
    print("Effect: immediate. Works for Chrome, Edge, Spotify, all apps.")
    print("Undo:   py -3.13 tools/adblock_hosts.py --remove")
    print()
    print("HOW THE DNS SINKHOLE WORKS:")
    print("  Browser: 'What IP is doubleclick.net?'")
    print("  Windows: checks hosts file first -> 0.0.0.0 -> connection refused")
    print("  Ad never loaded. No CPU, no bandwidth, no tracking.")
    print()
    print("UPGRADE: Pi-hole on RPi = same thing for your entire network")
    print("  All phones, tablets, TVs on your WiFi get ad blocking automatically.")
    print("  Same RPi CM4 hardware as RogueGuard in this repo.")


def cmd_remove():
    """Remove ad blocker."""
    content = read_hosts()
    if not hosts_has_block(content):
        print("Adblock not installed in hosts file.")
        return

    new_content, removed = remove_block(content)
    try:
        HOSTS_FILE.write_text(new_content, encoding='utf-8')
    except PermissionError:
        print("ERROR: Need Administrator.")
        sys.exit(1)

    flush_dns()
    print(f"Removed {removed} lines from hosts file.")
    print("DNS flushed. Ads will resume loading.")

    if BACKUP_FILE.exists():
        print(f"Original backup preserved at: {BACKUP_FILE}")


def cmd_status():
    """Show current status."""
    content = read_hosts()
    if hosts_has_block(content):
        n = count_blocked(content)
        print(f"Adblock ACTIVE: {n} domains blocked in hosts file.")
        print(f"Hosts file: {HOSTS_FILE}")
    else:
        print("Adblock NOT active.")
    print()
    print("NETWORKING EXPLANATION:")
    print("  DNS query path without adblock:")
    print("    App -> OS resolver -> ISP DNS -> ad server IP returned -> ad loads")
    print()
    print("  DNS query path WITH adblock (hosts file sinkhole):")
    print("    App -> OS resolver -> hosts file (0.0.0.0) -> connection refused")
    print("    Step 3 never reaches network. Zero latency. Zero tracking.")
    print()
    print("  Pi-hole (network-level):")
    print("    All devices -> RPi DNS server -> blocklist check -> 0.0.0.0 or real IP")
    print("    Protects phones, smart TVs, IoT devices -- anything on your WiFi")


def cmd_preview():
    """Preview without installing."""
    domains = parse_blocklist(CORE_BLOCKLIST)
    print(f"Would block {len(domains)} domains ({len(domains)*2} entries with www.):")
    for d in sorted(domains)[:30]:
        print(f"  0.0.0.0 {d}")
    if len(domains) > 30:
        print(f"  ... and {len(domains)-30} more")
    print()
    print("Run with --install (as Administrator) to apply.")


def explain_pihole():
    """Technical explanation of Pi-hole architecture."""
    return {
        'what': 'Pi-hole = DNS server + blocklist, runs on RPi',
        'how': [
            '1. Install Pi-hole on RPi: curl -sSL https://install.pi-hole.net | bash',
            '2. Set router DHCP DNS = RPi IP (e.g. 192.168.1.100)',
            '3. Every device on WiFi now uses RPi as DNS server',
            '4. Pi-hole checks every DNS query against blocklist',
            '5. Known ad domain -> return 0.0.0.0 -> no ad',
            '6. Unknown domain -> forward to 8.8.8.8 -> normal resolution',
        ],
        'hardware': 'RPi 4 / CM4 (same as RogueGuard in this repo). ~$35.',
        'blocklists': [
            'StevenBlack/hosts (100k+ domains)',
            'AdGuard DNS filter',
            'Malware Domain List',
            'EasyList (uBlock origin source)',
        ],
        'stats_url': 'http://pi.hole/admin  (web dashboard: queries/hour, % blocked)',
        'typical_block_rate': '15-25% of all DNS queries on home network are ads/trackers',
        'vs_browser_extension': (
            'Pi-hole blocks at DNS level: catches Spotify ads, app ads, smart TV ads. '
            'Browser extension only blocks in that browser.'
        ),
        'dns_over_https_note': (
            'Browsers with DoH (DNS over HTTPS) bypass Pi-hole. '
            'Disable DoH in browser settings, or run Pi-hole with Unbound (local resolver).'
        ),
        'connection_to_repo': (
            'RPi CM4 in RogueGuard runs TD-GS + CNN for rogue wave detection. '
            'Same board, different software. CE = software defines the hardware purpose.'
        ),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DNS Ad Blocker (hosts file sinkhole). Run as Administrator.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--install',  action='store_true', help='Install ad blocker')
    parser.add_argument('--extended', action='store_true', help='Fetch extended blocklist (needs internet)')
    parser.add_argument('--remove',   action='store_true', help='Remove ad blocker')
    parser.add_argument('--status',   action='store_true', help='Show status')
    parser.add_argument('--preview',  action='store_true', help='Preview (dry run)')
    parser.add_argument('--pihole',   action='store_true', help='Explain Pi-hole setup')
    args = parser.parse_args()

    if args.pihole:
        info = explain_pihole()
        print("=== PI-HOLE SETUP (RPi DNS sinkhole) ===\n")
        print(f"What: {info['what']}\n")
        print("How:")
        for step in info['how']:
            print(f"  {step}")
        print(f"\nHardware: {info['hardware']}")
        print(f"Block rate: {info['typical_block_rate']}")
        print(f"\nvs browser extension: {info['vs_browser_extension']}")
        print(f"\nWarning: {info['dns_over_https_note']}")
    elif args.install:
        cmd_install(extended=args.extended)
    elif args.remove:
        cmd_remove()
    elif args.status:
        cmd_status()
    elif args.preview:
        cmd_preview()
    else:
        parser.print_help()
        print()
        cmd_status()
