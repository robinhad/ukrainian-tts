# Repository instructions

- Do not commit local machine names, hostnames, usernames, absolute home or
  storage paths, credentials, tokens, private endpoints, or internal
  infrastructure information.
- Keep machine-specific configuration, scheduler logs, downloaded datasets,
  model weights, caches, and temporary runtime artifacts in ignored locations.
- Use repository-relative paths, environment variables, or explicit generic
  placeholders in committed code and documentation.
- Before each commit, inspect the staged diff for local or sensitive data.
  Publish only portable code, reproducible configuration, and sanitized findings.
- Commit and push tested progress regularly as `Codex <codex@openai.com>` when
  working on the authorized training iteration.
