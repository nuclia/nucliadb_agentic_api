# NucliaDB Agentic API

The NucliaDB Agentic API package exposes NucliaDB-oriented agentic capabilities
for Hyperforge, including ASK/search flows and MCP integrations.

## Install

From the workspace root:

```bash
uv sync
```

## Run

Start the service:

```bash
uv run nucliadb-agentic-api
```

Useful endpoints:

- `/health/ready`
- `/health/alive`
- `/metrics`

## Configuration

Runtime configuration is provided through environment variables consumed by
Pydantic settings. Common settings include:

- `HTTP_HOST` and `HTTP_PORT`
- `MEMORY_READER_NUCLIADB`, `MEMORY_WRITER_NUCLIADB`, and
  `MEMORY_SEARCH_NUCLIADB`
- `MEMORY_APIKEY_NUCLIADB`
- `VALKEY_URL`
- `IDP_REGIONAL_GRPC`
- `LOAD_MODULES`

## MCP sources with OAuth

Create an MCP source using `POST /api/v1/kb/{kbid}/sources/{source_id}`:

```json
{
  "type": "mcp",
  "uri": "https://mcp.example.com/mcp",
  "description": "Tools requiring OAuth",
  "auth_server_url": "https://auth.example.com",
  "client_id": "registered-client",
  "client_secret": "replace-with-client-secret",
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "scope": "read write",
  "pkce": true
}
```

Reference the source ID in `smart_agent.sources` when creating an agentic
configuration. The source endpoints support reading, listing, updating and
deleting OAuth sources in the same way as other sources. `PATCH` replaces the
source configuration; send the complete source payload, including any OAuth
fields that should be retained. The exception is `client_secret`: omitting it
preserves the stored secret, sending a new value replaces it, and sending
`null` removes it.

### Frontend API contract

All OAuth fields are optional and nullable. Omitted or null values leave the MCP
driver defaults unchanged; existing non-OAuth sources and their responses are
unchanged. Explicit `false`, empty lists and empty scopes are preserved.

| Field | JSON type | Behavior when omitted or null |
| --- | --- | --- |
| `auth_server_url` | string | OAuth disabled. Set this URL to enable OAuth. |
| `client_id` | string | Use dynamic client registration. |
| `client_secret` | string | No pre-registered client secret. |
| `scope` | string | Driver default: `"user"`. Separate multiple scopes with spaces. |
| `grant_types` | string array | Driver default: `[]`. For authorization code flows, send `["authorization_code", "refresh_token"]`. |
| `response_types` | string array | Driver default: `[]`. For authorization code flows, send `["code"]`. |
| `authorization_endpoint` | string | Discover the authorization endpoint from OAuth metadata. |
| `token_endpoint` | string | Discover the token endpoint from OAuth metadata. |
| `pkce` | boolean | Driver default: `true`. |
| `redirect_uris` | string array | Server-managed; the driver overrides supplied values with `MCP_CALLBACK_URL`. |

These fields are described in the `MCPSourceSchema` OpenAPI component.
The frontend should offer an optional OAuth section and should not require users
to edit redirect URIs. For pre-registered clients, register the backend callback
URL with the OAuth provider. Without `client_id`, the provider must support
dynamic client registration.

`client_secret` is write-only: GET/list responses never return it, including
to owners. It is encrypted at rest using Hyperforge's encryption helpers and
`ENCRYPTION_SECRET_KEY`, and decrypted internally when loading the source for
execution. The API and agentic server must share the same encryption key.
When editing a source, leave the secret field out unless the user replaces or
removes it. Do not log source payloads or expose credentials in client-side
diagnostics.

## Development

Run the package tests from the workspace root:

```bash
make test
```

Docker is required. The S3 fixtures pin LocalStack to `0.14.5` because the
dependency's `0.12.18` image is no longer available from Docker Hub.

# Objective
Enhance the RAG experience by offering RAO features directly in a KB.

# How?
The same way we store search configs in the KB, we could store agentic configs, and when calling the `/ask` endpoint we can refer to a given agentic config.

# Scope

The corresponding RAO workflow will be:
- a Rephrase (optionally)
- a SmartAgent
- a Summarize

Possible sources for the SmartAgent:
- The current KB (possibly several times with different filters)
- Sync service (using connections defined in the current KB)
- Google
- Perplexity
- MCP

Out of scope
- Other KBs
- SQL
- Snowflake (?)
- Sitefinity (?)

# Config spec

```json
{
  "rephrase": { // optional
    "ask_to": <filter_expression>, //optional
    "prompt": <string>, // optional
    "model": <model>, // optional
  },
  "smart_agent": {
    "mode": <reactive | plan_execute>,
    "extra_prompt": <string>, // optional
    "models": { // optional
      "context_validation": <model>, // optional
      "planner": <model>, // optional
      "executor": <model>, // optional
    },
    "sources": [<source list>]
  },
  "summarize": { // optional
    "user_prompt": <string>, // optional
    "system_prompt": <string>, // optional
    "conversational": <boolean>,
    "model": <model>, // optional
    // and citations must be forced to chunk-level
  }
}
```

And sources can be:

```json
{
  "type": "nucliadb",
  "description": <string>,
  "search_config": <ask search configuration id>, // optional
  "filter_expression": <filter_expression>, //optional
}

{
  "type": "sync",
  "description": <string>,
  "connection": <connection_id>
}

{
  "type": "google",
  "description": <string>
}

{
  "type": "perplexity",
  "description": <string>
}

{
  "type": "mcp",
  "description": <string>,
  <...the MCP driver params>
}
````
