"""Add sources table

Revision ID: b3c4d5e6f7a8
Revises: 7f3b2f92a8c1
Create Date: 2026-06-08 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "7f3b2f92a8c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sources_table",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("kbid", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("agentic_config_id", sa.String(), nullable=True),
        sa.Column("created", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["account", "kbid", "agentic_config_id"],
            [
                "agentic_config_table.account",
                "agentic_config_table.kbid",
                "agentic_config_table.agentic_id",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("account", "kbid", "source_id"),
    )
    op.create_index(
        op.f("ix_sources_table_account"),
        "sources_table",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_sources_table_kbid"),
        "sources_table",
        ["kbid"],
        unique=False,
    )
    op.create_index(
        op.f("ix_sources_table_agentic_config_id"),
        "sources_table",
        ["agentic_config_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_sources_table_agentic_config_id"), table_name="sources_table"
    )
    op.drop_index(op.f("ix_sources_table_kbid"), table_name="sources_table")
    op.drop_index(op.f("ix_sources_table_account"), table_name="sources_table")
    op.drop_table("sources_table")
