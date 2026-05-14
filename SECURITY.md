# Security Policy

## Reporting a Vulnerability

Please use GitHub private vulnerability reporting for security issues in this
repository when it is available:

https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/security/advisories/new

Do not open a public issue with exploit details, proof-of-concept payloads, or
private environment information. If private reporting is unavailable, open a
public issue asking for a preferred security contact without including the
technical details.

Useful reports include:

- affected version or commit
- operating system and Python version
- whether the browser editor, CLI, or Python API is involved
- concise reproduction steps
- expected impact, if known

## Maintainer Disclosure Checklist

For a confirmed issue:

1. Prepare the fix privately or in a normal pull request when the details are
   already public.
2. Publish the patched release before publishing the advisory, unless users
   need immediate mitigation guidance.
3. Create or update a GitHub Security Advisory with affected versions, patched
   versions, severity, impact, workarounds, and references.
4. Mention the fix in `CHANGELOG.md` and the release notes.
5. Consider yanking affected PyPI releases only when discouraging new installs
   of those exact releases is safer than leaving them available. Prefer a clear
   yank reason that points users to the patched release.

In short: publish the patched release before publishing the advisory when users
do not need immediate mitigation guidance.

## PrismJS Advisory Draft

Use this when publishing the bundled PrismJS update as a repository advisory
for `tensor-network-editor`. This is not a new PrismJS vulnerability; it is a
vendored dependency advisory that points to the upstream issue.

- Title: Bundled PrismJS before 1.30.0 in the browser-based editor
- Related upstream advisory: CVE-2024-53382 / GHSA-x7hr-w5r2-h6wg
- Affected package: `tensor-network-editor`
- Affected versions: releases that bundle PrismJS 1.29.0 in
  `src/tensor_network_editor/app/static/vendor/`
- Patched version: the first release that bundles PrismJS 1.30.0 or later
- Severity: Moderate, matching the upstream PrismJS advisory unless new project
  evidence shows a different impact

Suggested impact text:

```text
Tensor Network Editor bundled PrismJS 1.29.0 for syntax highlighting in the
browser-based editor. PrismJS versions before 1.30.0 are affected by
CVE-2024-53382 / GHSA-x7hr-w5r2-h6wg.

Installing or importing the Python package alone does not execute PrismJS. The
affected code path is the browser-based editor. Risk is higher if the local
editor is exposed beyond localhost, or if untrusted HTML-like content can reach
the editor UI.
```

Suggested recommendation text:

```text
Upgrade to the patched Tensor Network Editor release. If you cannot upgrade
immediately, avoid exposing the local editor outside trusted loopback/local
workflows and avoid opening untrusted designs or Python-derived content in the
browser editor.
```
