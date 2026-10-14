# CHANGELOG

## Unreleased

- `ask()` accepts `forwarded_headers` (e.g. `x-allow-access-external-models`),
  which are sent on every learning call it makes. The NucliaDB agent forwards them
  from the question headers.

## 1.0.0

- Initial version
