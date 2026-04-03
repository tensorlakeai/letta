"""support tensorlake sandbox type

Revision ID: a1b2c3d4e5f9
Revises: 1c28e167b74f
Create Date: 2026-03-30 00:00:00.000000

"""

from typing import Sequence, Union

from sqlalchemy import text

from alembic import op
from letta.settings import DatabaseChoice, settings

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f9"
down_revision: Union[str, None] = "1c28e167b74f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite uses plain strings — no enum type to alter.
    if settings.database_engine == DatabaseChoice.POSTGRES:
        op.execute("ALTER TYPE sandboxtype ADD VALUE 'TENSORLAKE' AFTER 'MODAL'")


def downgrade() -> None:
    if settings.database_engine == DatabaseChoice.POSTGRES:
        connection = op.get_bind()

        data_conflicts = connection.execute(
            text(
                """
            SELECT COUNT(*)
            FROM sandbox_configs
            WHERE "type" NOT IN ('E2B', 'LOCAL', 'MODAL')
        """
            )
        ).fetchone()
        if data_conflicts[0]:
            raise RuntimeError(
                (
                    "Cannot downgrade enum: Data conflicts are detected in sandbox_configs.sandboxtype.\n"
                    "Please manually handle these records before handling the downgrades.\n"
                    f"{data_conflicts} invalid sandboxtype values"
                )
            )

        # Postgres does not support dropping enum values — create a replacement enum and swap.
        op.execute("CREATE TYPE sandboxtype_old AS ENUM ('E2B', 'LOCAL', 'MODAL')")
        op.execute('ALTER TABLE sandbox_configs ALTER COLUMN "type" TYPE sandboxtype_old USING "type"::text::sandboxtype_old')
        op.execute("DROP TYPE sandboxtype")
        op.execute("ALTER TYPE sandboxtype_old RENAME to sandboxtype")
