"""Add flexible highlight system tables

Revision ID: 003_flexible_highlights
Revises: 002
Create Date: 2024-01-30

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "003_flexible_highlights"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add dimension_sets and highlight_type_registries tables."""

    # Create dimension_sets table
    op.create_table(
        "dimension_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("dimensions", sa.JSON(), nullable=False),
        sa.Column("dimension_weights", sa.JSON(), nullable=False),
        sa.Column("allow_partial_scoring", sa.Boolean(), nullable=True),
        sa.Column("minimum_dimensions_required", sa.Integer(), nullable=True),
        sa.Column("weight_normalization", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id", "name", name="uq_dimension_set_org_name"
        ),
    )
    op.create_index(
        op.f("ix_dimension_sets_organization_id"),
        "dimension_sets",
        ["organization_id"],
        unique=False,
    )

    # Create highlight_type_registries table
    op.create_table(
        "highlight_type_registries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("types", sa.JSON(), nullable=False),
        sa.Column("allow_multiple_types", sa.Boolean(), nullable=True),
        sa.Column("max_types_per_highlight", sa.Integer(), nullable=True),
        sa.Column("include_built_in_types", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", name="uq_type_registry_org"),
    )
    op.create_index(
        op.f("ix_highlight_type_registries_organization_id"),
        "highlight_type_registries",
        ["organization_id"],
        unique=True,
    )

    # Add new columns to streams table for flexible processing
    op.add_column("streams", sa.Column("dimension_set_id", sa.Integer(), nullable=True))
    op.add_column("streams", sa.Column("type_registry_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_streams_dimension_set",
        "streams",
        "dimension_sets",
        ["dimension_set_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_streams_type_registry",
        "streams",
        "highlight_type_registries",
        ["type_registry_id"],
        ["id"],
    )

    # Add new columns to batches table for flexible processing
    op.add_column("batches", sa.Column("dimension_set_id", sa.Integer(), nullable=True))
    op.add_column("batches", sa.Column("type_registry_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_batches_dimension_set",
        "batches",
        "dimension_sets",
        ["dimension_set_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_batches_type_registry",
        "batches",
        "highlight_type_registries",
        ["type_registry_id"],
        ["id"],
    )

    # Remove old highlight_type column and add new highlight_types JSON column
    op.drop_column("highlights", "highlight_type")
    op.add_column(
        "highlights",
        sa.Column("highlight_types", sa.JSON(), nullable=False, server_default="[]"),
    )

    # Create a trigger to update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    op.execute("""
        CREATE TRIGGER update_dimension_sets_updated_at BEFORE UPDATE
        ON dimension_sets FOR EACH ROW
        EXECUTE PROCEDURE update_updated_at_column();
    """)

    op.execute("""
        CREATE TRIGGER update_highlight_type_registries_updated_at BEFORE UPDATE
        ON highlight_type_registries FOR EACH ROW
        EXECUTE PROCEDURE update_updated_at_column();
    """)


def downgrade() -> None:
    """Remove flexible highlight system tables."""

    # Drop triggers
    op.execute(
        "DROP TRIGGER IF EXISTS update_dimension_sets_updated_at ON dimension_sets"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS update_highlight_type_registries_updated_at ON highlight_type_registries"
    )

    # Restore old highlight_type column and remove new one
    op.add_column(
        "highlights",
        sa.Column(
            "highlight_type", sa.String(50), nullable=False, server_default="custom"
        ),
    )
    op.drop_column("highlights", "highlight_types")

    # Remove foreign keys and columns from batches
    op.drop_constraint("fk_batches_type_registry", "batches", type_="foreignkey")
    op.drop_constraint("fk_batches_dimension_set", "batches", type_="foreignkey")
    op.drop_column("batches", "type_registry_id")
    op.drop_column("batches", "dimension_set_id")

    # Remove foreign keys and columns from streams
    op.drop_constraint("fk_streams_type_registry", "streams", type_="foreignkey")
    op.drop_constraint("fk_streams_dimension_set", "streams", type_="foreignkey")
    op.drop_column("streams", "type_registry_id")
    op.drop_column("streams", "dimension_set_id")

    # Drop tables
    op.drop_index(
        op.f("ix_highlight_type_registries_organization_id"),
        table_name="highlight_type_registries",
    )
    op.drop_table("highlight_type_registries")

    op.drop_index(
        op.f("ix_dimension_sets_organization_id"), table_name="dimension_sets"
    )
    op.drop_table("dimension_sets")
