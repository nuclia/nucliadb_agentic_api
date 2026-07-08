from hyperforge.server.settings import Settings as HyperforgeServerSettings

MODULES = [
    "hyperforge_rephrase",
    "hyperforge_nucliadb",
    "hyperforge_nucliadb_agentic",
    "hyperforge_summarize",
    "hyperforge_smart",
    "hyperforge_mcp",
    "hyperforge_google",
    "hyperforge_perplexity",
]


class Settings(HyperforgeServerSettings):
    load_modules: list[str] = MODULES
