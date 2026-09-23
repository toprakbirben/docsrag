import typer

from stuffrag import db

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer()
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init() -> None:
    """Create the schema (idempotent)."""
    tables = db.init()
    typer.echo(f"tables: {', '.join(tables)}")
