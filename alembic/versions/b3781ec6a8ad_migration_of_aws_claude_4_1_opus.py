"""migration of aws claude 4 1 opus

Revision ID: b3781ec6a8ad
Revises: 0fe6c7b5f33d
Create Date: 2026-10-01 13:04:16.465758

"""

from typing import Sequence, Union

from alembic import op
from nucliadb_agentic_api.db.llm_migration_utils import migrate_llm_models

# revision identifiers, used by Alembic.
revision: str = "b3781ec6a8ad"
down_revision: Union[str, Sequence[str], None] = "0fe6c7b5f33d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    migrate_llm_models(op.get_bind(), {"aws-claude-4-1-opus": "aws-claude-4-8-opus"})


def downgrade() -> None:
    pass
