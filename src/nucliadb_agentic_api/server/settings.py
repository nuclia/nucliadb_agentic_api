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
    activate_subject: str = "ndb_agentic.activate"
    answers_subject: str = (
        "ndb_agentic.{account}.{agent_id}.{workflow_id}.{session}.{question}.answer"
    )
    oauth_subject: str = "ndb_agentic.{account}.{agent_id}.{workflow_id}.{session}.{question}.oauth.{oauth_uuid}"
