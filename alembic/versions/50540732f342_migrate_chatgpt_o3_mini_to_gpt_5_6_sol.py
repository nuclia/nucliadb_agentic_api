"""migrate chatgpt o3 mini to gpt 5 6 sol

Revision ID: 50540732f342
Revises: b3781ec6a8ad
Create Date: 2026-11-05 17:12:25.297473

"""

from typing import Sequence, Union

from alembic import op
from nucliadb_agentic_api.db.llm_migration_utils import migrate_llm_models

# revision identifiers, used by Alembic.
revision: str = "50540732f342"
down_revision: Union[str, Sequence[str], None] = "b3781ec6a8ad"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    migrate_llm_models(op.get_bind(), {"chatgpt-o3-mini": "chatgpt-5.6-sol"})


def downgrade() -> None:
    """Downgrade schema."""
    pass
