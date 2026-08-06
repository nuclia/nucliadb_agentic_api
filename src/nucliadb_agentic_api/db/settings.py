from pydantic_settings import BaseSettings


class DataManagerSettings(BaseSettings):
    postgresql_dsn: str
    agentic_config_deletion_retention_days: int = 30
    export_read_chunk_size: int = 1024 * 1024  # 1 MB
    export_read_max_size: int = 10 * 1024 * 1024  # 10 MB

    hyperforge_google_credentials: str = ""
    hyperforge_google_location: str = ""
    hyperforge_perplexity_key: str = ""
