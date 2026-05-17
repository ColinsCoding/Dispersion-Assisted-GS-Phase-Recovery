#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# update-sri.sh - rewrite the integrity="..." attributes in web/index.html
#                 with the real SHA-384 digests of the currently-pinned URLs.
#
# Usage:  ./tools/update-sri.sh
#
# Requires: bash, curl, openssl, sed.  Runs locally - it does not change
# anything in the repo besides web/index.html.
# -----------------------------------------------------------------------------
set -euo pipefail

HTML="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/web/index.html"
if [[ ! -f "$HTML" ]]; then
  echo "web/index.html not found - run from the repo root" >&2
  exit 1
fi

# Extract every (src, integrity-marker) pair we know about.
declare -A urls
urls[PLOTLY]='https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.32.0/plotly.min.js'

for key in "${!urls[@]}"; do
  url="${urls[$key]}"
  echo "fetching $url"
  hash=$(curl -fsSL "$url" | openssl dgst -sha384 -binary | openssl base64 -A)
  if [[ -z "$hash" ]]; then
    echo "failed to compute hash for $url" >&2; exit 1
  fi
  # In-place sed: macOS uses -i '' while GNU sed uses -i, this works on both.
  sed -i.bak -E "s#sha384-INTEGRITY-TODO-${key}#sha384-${hash}#g" "$HTML"
  rm -f "${HTML}.bak"
  echo "  $key  sha384-${hash}"
done

echo
echo "Updated $HTML.  Commit with:"
echo "  git add web/index.html && git commit -m 'web: refresh SRI hashes'"
