"""migrate sonnet and gpt 5 6 models

Revision ID: 0fe6c7b5f33d
Revises: c4d5e6f7a8b9
Create Date: 2026-09-21 10:28:51.980835

"""

from typing import Sequence, Union

from alembic import op
from nucliadb_agentic_api.db.llm_migration_utils import migrate_llm_models

# revision identifiers, used by Alembic.
revision: str = "0fe6c7b5f33d"
down_revision: Union[str, Sequence[str], None] = "c4d5e6f7a8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    migrate_llm_models(
        op.get_bind(),
        {
            "claude-4-5-sonnet": "claude-5-sonnet",
            "chatgpt-azure-4o": "chatgpt-azure-5.6-terra",
            "chatgpt-azure-4o-mini": "chatgpt-azure-5.6-luna",
            "chatgpt-azure-o3-mini": "chatgpt-azure-5.6-luna",
        },
    )


def downgrade() -> None:
    """Downgrade schema."""
    pass
