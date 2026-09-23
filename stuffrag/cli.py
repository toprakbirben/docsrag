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


@app.command()
def sync(source: str) -> None:
    """Ingest a source into the documents table. Sources: fastapi."""
    if source != "fastapi":
        raise typer.BadParameter(f"unknown source {source!r}; available: fastapi")
    from stuffrag.ingest import fastapi_repo

    repo = fastapi_repo.checkout()
    docs, missing = fastapi_repo.collect(repo, fastapi_repo.FASTAPI_TAG)
    for m in missing:
        typer.echo(f"warning: unresolved include {m}", err=True)
    with db.connect() as conn:
        new, changed = db.upsert_documents(conn, docs)
    typer.echo(f"fastapi {fastapi_repo.FASTAPI_TAG}: {len(docs)} docs ({new} new, {changed} changed), "
               f"{len(missing)} unresolved includes")
