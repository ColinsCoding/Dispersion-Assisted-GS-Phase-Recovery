# Security policy

This repository is a research / coursework project, not a production service.
Even so, the static viewer under `web/` is written to fail closed and to make
the threat model explicit.

## Threat model

The static page in `web/` is intended to be served either from GitHub Pages
or from a local file server. It does not include any backend, accepts no
user input that is sent to a remote service, and does not handle credentials
or PII. The only data it reads is `web/data/sample.json` from the same
origin.

Risks we explicitly try to prevent:

- **Compromised CDN script** — pinned versions plus Subresource Integrity (SRI)
  attributes on every `<script src="https://...">`. The browser refuses to run
  a script whose hash doesn't match.
- **Inline-script injection** — no `'unsafe-inline'` in the `script-src` CSP
  directive. All JS lives under `web/js/` and is loaded as a module.
- **Clickjacking** — `frame-ancestors 'none'` in the CSP plus
  `X-Frame-Options: DENY` (set by GitHub Pages by default).
- **Open redirects / cross-origin form posts** — `form-action 'self'` and
  `base-uri 'self'`.
- **Data exfiltration via referrer** — `<meta name="referrer" content="no-referrer">`.

## Updating Subresource Integrity hashes

The `<script>` tags in `web/index.html` start with placeholder
`integrity="sha384-INTEGRITY-TODO-…"` values. The browser will refuse to run
the CDN bundles until they are populated. After bumping a pinned version run

```sh
./tools/update-sri.sh
```

which fetches each referenced URL, computes the SHA-384 digest, and rewrites
the `integrity` attribute in `web/index.html` in place.

## Reporting a vulnerability

Open a GitHub issue with the `security` label, or — for anything you would
rather not publish — email the repository owner. Please do not file a public
issue with reproducer details for an unpatched vulnerability.

## Out of scope

Network-level concerns (transport, certificates, hosting platform misconfig)
are the responsibility of whoever deploys this. If you host a fork, audit your
deployment.
