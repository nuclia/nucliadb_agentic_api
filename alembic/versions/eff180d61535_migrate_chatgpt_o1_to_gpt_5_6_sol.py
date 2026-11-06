"""migrate chatgpt o1 to gpt 5 6 sol

Revision ID: eff180d61535
Revises: 50540732f342
Create Date: 2026-11-06 15:29:58.883376

"""

from typing import Sequence, Union

from alembic import op
from nucliadb_agentic_api.db.llm_migration_utils import migrate_llm_models

# revision identifiers, used by Alembic.
revision: str = "eff180d61535"
down_revision: Union[str, Sequence[str], None] = "50540732f342"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    migrate_llm_models(op.get_bind(), {"chatgpt-o1": "chatgpt-5.6-sol"})


def downgrade() -> None:
    """Downgrade schema."""
    pass
