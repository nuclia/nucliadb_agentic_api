"""retire Azure o series models

Revision ID: 62c817bd3550
Revises: eff180d61535
Create Date: 2026-11-16 09:35:20.661977

"""

from typing import Sequence, Union

from alembic import op
from nucliadb_agentic_api.db.llm_migration_utils import migrate_llm_models

# revision identifiers, used by Alembic.
revision: str = "62c817bd3550"
down_revision: Union[str, Sequence[str], None] = "eff180d61535"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    migrate_llm_models(
        op.get_bind(),
        {
            "chatgpt-azure-o1": "chatgpt-azure-5.6-sol",
            "chatgpt-azure-o3": "chatgpt-azure-5.6-sol",
            "chatgpt-azure-o4-mini": "chatgpt-azure-5.6-terra",
        },
    )


def downgrade() -> None:
    """Downgrade schema."""
    pass
