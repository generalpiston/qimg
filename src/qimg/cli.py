from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from qimg.config import default_config_path, load_config, write_default_config
from qimg.mcp.server_http import run_http_server
from qimg.mcp.server_stdio import run_stdio_server
from qimg.model_manager import get_model_manager, result_to_dict
from qimg.service import QimgService, SearchFilters
from qimg.util.logging import setup_logging, use_color

app = typer.Typer(help="qimg: Query Images")
collection_app = typer.Typer(help="Manage collections")
context_app = typer.Typer(help="Manage contexts")
facts_app = typer.Typer(help="Manage extracted facts")
models_app = typer.Typer(help="Manage vendored model assets")
app.add_typer(collection_app, name="collection")
app.add_typer(collection_app, name="collections")
app.add_typer(context_app, name="context")
app.add_typer(facts_app, name="facts")
app.add_typer(models_app, name="model")
app.add_typer(models_app, name="models")


@dataclass(slots=True)
class AppState:
    service: QimgService
    console: Console
    config_path: Path


def _print_logo(console: Console, show_logo: bool) -> None:
    if not show_logo:
        return
    logo = (
        "  ██████      ██   ███    ███    ██████   \n"
        " ██    ██     ██   ████  ████   ██         \n"
        " ██    ██     ██   ██ ████ ██   ██   ███   \n"
        " ██ ▄▄ ██     ██   ██  ██  ██   ██    ██   \n"
        "  ██████  ██  ██   ██      ██    ██████    \n"
        "    ▀▀    ▀▀  ▀▀   ▀▀      ▀▀      ▀▀▀▀     "
    )
    console.print()
    console.print(f"[bold cyan]{logo}[/bold cyan]")
    console.print("[dim]contexts • facts • vectors • metadata[/dim]")
    console.print()


def _state(ctx: typer.Context) -> AppState:
    st = ctx.obj
    if not isinstance(st, AppState):
        raise RuntimeError("app state not initialized")
    return st


def _context_texts(row: dict[str, Any]) -> list[str]:
    contexts = row.get("contexts") or []
    out: list[str] = []
    for item in contexts:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                out.append(text)
        elif isinstance(item, str):
            t = item.strip()
            if t:
                out.append(t)
    return out


def _emit_results(console: Console, rows: list[dict], json_out: bool, files_out: bool, debug: bool = False) -> None:
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return

    if files_out:
        for row in rows:
            contexts = "|".join(_context_texts(row))
            md = row.get("metadata") or {}
            dt = md.get("date", "") if isinstance(md, dict) else ""
            score = (row.get("scores") or {}).get("overall", 0.0)
            typer.echo(
                "\t".join(
                    [
                        str(row.get("id", "")),
                        f"{float(score):.4f}",
                        str(row.get("path", "")),
                        contexts,
                        str(md.get("width") or ""),
                        str(md.get("height") or ""),
                        str(dt),
                    ]
                )
            )
        return

    if not rows:
        console.print("[dim]no results[/dim]")
        return

    for idx, row in enumerate(rows, start=1):
        scores = row.get("scores") or {}
        md = row.get("metadata") or {}
        facts = row.get("facts") or {}

        console.print(
            f"[bold cyan]{idx}.[/bold cyan] "
            f"[bold magenta]{row.get('id', '')}[/bold magenta]  "
            f"[bold green]{float(scores.get('overall', 0.0)):.3f}[/bold green]  "
            f"[white]{row.get('path', '')}[/white]"
        )

        contexts = _context_texts(row)
        if contexts:
            console.print(f"   [bold blue]Context:[/bold blue] {' | '.join(contexts[:2])}")
        else:
            console.print("   [bold blue]Context:[/bold blue] [dim](none)[/dim]")

        caption = (facts.get("caption") if isinstance(facts, dict) else None) or ""
        if caption:
            console.print(f"   [bold yellow]Caption:[/bold yellow] {caption}")
        else:
            console.print("   [bold yellow]Caption:[/bold yellow] [dim](no caption; run qimg facts extract --caption)[/dim]")

        date = md.get("date") if isinstance(md, dict) else None
        camera = md.get("camera") if isinstance(md, dict) else None
        if date or camera:
            parts = [str(x) for x in [date, camera] if x]
            console.print(f"   [bold]EXIF:[/bold] {' | '.join(parts)}")

        if debug:
            console.print(
                "   [dim]Debug: "
                f"overall={float(scores.get('overall', 0.0)):.4f} "
                f"rrf={scores.get('rrf')} "
                f"lex_total={scores.get('lexical_total')} "
                f"lex_nf={scores.get('lexical_namefolder')} "
                f"lex_ctx={scores.get('lexical_context')} "
                f"lex_facts={scores.get('lexical_facts')} "
                f"vector={scores.get('vector')} "
                f"rerank={scores.get('rerank')}"
                "[/dim]"
            )
            contrib = scores.get("contributions") or []
            if contrib:
                console.print(f"   [dim]Channels: {', '.join(str(x) for x in contrib)}[/dim]")


def _emit_obj(console: Console, obj: dict, json_out: bool) -> None:
    if json_out:
        typer.echo(json.dumps(obj, indent=2))
        return
    for k, v in obj.items():
        console.print(f"[bold]{k}[/bold]: {v}")


def _filters(
    num: int,
    collection: str | None,
    path_prefix: str | None,
    after: str | None,
    before: str | None,
    min_width: int | None,
    min_height: int | None,
    min_score: float,
    all_results: bool,
) -> SearchFilters:
    return SearchFilters(
        num=num,
        collection=collection,
        path_prefix=path_prefix,
        after=after,
        before=before,
        min_width=min_width,
        min_height=min_height,
        min_score=min_score,
        all_results=all_results,
    )


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="Config YAML path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging")] = False,
) -> None:
    setup_logging(verbose)
    cfg_path = config.expanduser() if config else default_config_path()
    if not cfg_path.exists():
        write_default_config(cfg_path)
    cfg = load_config(cfg_path)
    svc = QimgService(cfg)
    color_on = use_color()
    console = Console(color_system="auto" if color_on else None, force_terminal=color_on)
    _print_logo(console, show_logo=bool(getattr(cfg.ui, "show_logo", True)))
    ctx.obj = AppState(
        service=svc,
        console=console,
        config_path=cfg_path,
    )


@app.command("init-config")
def init_config(
    ctx: typer.Context,
    path: Annotated[Path | None, typer.Option("--path", help="Write config to this path")] = None,
    download_models: Annotated[bool, typer.Option("--download-models", help="Download vendored models after writing config")] = False,
    image: Annotated[bool, typer.Option("--image", help="Download image encoder model")] = False,
    text: Annotated[bool, typer.Option("--text", help="Download text encoder model")] = False,
    obj: Annotated[bool, typer.Option("--object", help="Download object detector model")] = False,
    from_cache: Annotated[bool, typer.Option("--from-cache", help="Copy models from local HF cache instead of downloading")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    written = write_default_config(path.expanduser() if path else None)
    should_download = download_models or image or text or obj
    payload: dict[str, Any] = {"config_path": str(written), "downloaded": []}
    if should_download:
        try:
            results = get_model_manager().download_selected(
                download_image=image,
                download_text=text,
                download_object=obj,
                from_cache=from_cache,
            )
        except Exception as exc:
            st.console.print(f"[red]model download failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        payload["downloaded"] = [result_to_dict(item) for item in results]

    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return

    st.console.print(f"[green]config:[/green] {written}")
    downloaded = payload.get("downloaded") or []
    if not downloaded:
        return
    table = Table(title="downloaded models")
    table.add_column("kind")
    table.add_column("model")
    table.add_column("revision")
    table.add_column("base_dir")
    for item in downloaded:
        table.add_row(
            str(item.get("kind", "")),
            str(item.get("model_name", "")),
            str(item.get("revision", "")),
            str(item.get("base_dir", "")),
        )
    st.console.print(table)


@models_app.command("download")
def model_download_cmd(
    ctx: typer.Context,
    image: Annotated[bool, typer.Option("--image", help="Download image encoder model")] = False,
    text: Annotated[bool, typer.Option("--text", help="Download text encoder model")] = False,
    obj: Annotated[bool, typer.Option("--object", help="Download object detector model")] = False,
    from_cache: Annotated[bool, typer.Option("--from-cache", help="Copy models from local HF cache instead of downloading")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    try:
        results = get_model_manager().download_selected(
            download_image=image,
            download_text=text,
            download_object=obj,
            from_cache=from_cache,
        )
    except Exception as exc:
        st.console.print(f"[red]model download failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    rows = [result_to_dict(item) for item in results]
    if json_out:
        typer.echo(json.dumps({"downloaded": rows}, indent=2))
        return
    table = Table(title="downloaded models")
    table.add_column("kind")
    table.add_column("model")
    table.add_column("revision")
    table.add_column("base_dir")
    for row in rows:
        table.add_row(
            str(row.get("kind", "")),
            str(row.get("model_name", "")),
            str(row.get("revision", "")),
            str(row.get("base_dir", "")),
        )
    st.console.print(table)


@collection_app.command("add")
def collection_add(
    ctx: typer.Context,
    root_path: Annotated[str, typer.Argument(help="Root folder")],
    name: Annotated[str, typer.Option("--name", help="Collection name")],
    mask: Annotated[str, typer.Option("--mask", help="Glob-like mask")]
    = "**/*.{jpg,jpeg,png,webp,tif,tiff,heic}",
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    result = st.service.collection_add(root_path=root_path, name=name, mask=mask)
    _emit_obj(st.console, result, json_out)


@collection_app.command("list")
def collection_list_cmd(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.collection_list()
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return
    table = Table(title="collections")
    table.add_column("name")
    table.add_column("root_path")
    table.add_column("mask")
    for row in rows:
        table.add_row(str(row["name"]), str(row["root_path"]), str(row["mask"]))
    st.console.print(table)


@collection_app.command("ls")
def collection_ls_cmd(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    collection_list_cmd(ctx, json_out)


@collection_app.command("remove")
def collection_remove_cmd(ctx: typer.Context, name: str) -> None:
    st = _state(ctx)
    st.service.collection_remove(name)
    typer.echo(f"removed {name}")


@collection_app.command("rm")
def collection_rm_cmd(ctx: typer.Context, name: str) -> None:
    collection_remove_cmd(ctx, name)


@collection_app.command("rename")
def collection_rename_cmd(ctx: typer.Context, old_name: str, new_name: str) -> None:
    st = _state(ctx)
    st.service.collection_rename(old_name, new_name)
    typer.echo(f"renamed {old_name} -> {new_name}")


@context_app.command("add")
def context_add_cmd(
    ctx: typer.Context,
    virtual_path: str,
    text: str,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    result = st.service.context_add(virtual_path, text)
    _emit_obj(st.console, result, json_out)


@context_app.command("list")
def context_list_cmd(
    ctx: typer.Context,
    prefix: Annotated[str | None, typer.Option("--prefix")] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.context_list(prefix=prefix)
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return
    table = Table(title="contexts")
    table.add_column("virtual_path")
    table.add_column("text")
    for row in rows:
        table.add_row(str(row["virtual_path"]), str(row["text"]))
    st.console.print(table)


@context_app.command("rm")
def context_rm_cmd(ctx: typer.Context, virtual_path: str) -> None:
    st = _state(ctx)
    st.service.context_rm(virtual_path)
    typer.echo(f"removed {virtual_path}")


@facts_app.command("extract")
def facts_extract_cmd(
    ctx: typer.Context,
    identifiers: Annotated[list[str], typer.Argument()] = [],
    caption: Annotated[bool, typer.Option("--caption", help="Extract captions (defaults to on when no extract flags are passed)")] = False,
    tags: Annotated[bool, typer.Option("--tags", help="Extract tags (defaults to on when no extract flags are passed)")] = False,
    objects: Annotated[bool, typer.Option("--objects", help="Extract object labels (defaults to on when no extract flags are passed)")] = False,
    ocr: Annotated[bool, typer.Option("--ocr", help="Extract OCR text (defaults to on when no extract flags are passed)")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    result = st.service.facts_extract(
        extract_caption=caption,
        extract_tags=tags,
        extract_objects=objects,
        extract_ocr=ocr,
        identifiers=identifiers,
    )
    _emit_obj(st.console, result, json_out)


@facts_app.command("ls")
def facts_ls_cmd(
    ctx: typer.Context,
    identifier: Annotated[str | None, typer.Argument()] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.facts_ls(identifier)
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return
    title = f"facts: {identifier}" if identifier else "facts: all"
    table = Table(title=title)
    table.add_column("asset_id")
    table.add_column("rel_path")
    table.add_column("id")
    table.add_column("type")
    table.add_column("key")
    table.add_column("value")
    table.add_column("confidence")
    table.add_column("source")
    for row in rows:
        value = row.get("value_text") if row.get("value_text") else row.get("value_json")
        table.add_row(
            str(row.get("asset_id", "")),
            str(row.get("rel_path", "")),
            str(row.get("id", "")),
            str(row.get("fact_type", "")),
            str(row.get("key", "")),
            str(value or ""),
            str(row.get("confidence", "")),
            str(row.get("source", "")),
        )
    st.console.print(table)


@facts_app.command("rm")
def facts_rm_cmd(
    ctx: typer.Context,
    source: Annotated[str, typer.Option("--source", help="Source name to remove")],
    identifier: Annotated[str | None, typer.Option("--asset", help="Optional asset id/path")] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    result = st.service.facts_rm(source=source, identifier=identifier)
    _emit_obj(st.console, result, json_out)


@app.command("update")
def update_cmd(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    stats = st.service.update()
    _emit_obj(st.console, stats, json_out)


@app.command("embed")
def embed_cmd(
    ctx: typer.Context,
    images: Annotated[bool, typer.Option("--images", help="Embed images")] = False,
    ocr_text: Annotated[bool, typer.Option("--ocr-text", help="Embed OCR text")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    if images or ocr_text:
        include_images = images
        include_ocr_text = ocr_text
    else:
        include_images = True
        include_ocr_text = True
    stats = st.service.embed(include_images=include_images, include_ocr_text=include_ocr_text)
    _emit_obj(st.console, stats, json_out)


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    query: str,
    num: Annotated[int, typer.Option("-n", "--num")] = 10,
    collection: Annotated[str | None, typer.Option("-c", "--collection")] = None,
    path_prefix: Annotated[str | None, typer.Option("--path-prefix")] = None,
    after: Annotated[str | None, typer.Option("--after")] = None,
    before: Annotated[str | None, typer.Option("--before")] = None,
    min_width: Annotated[int | None, typer.Option("--min-width")] = None,
    min_height: Annotated[int | None, typer.Option("--min-height")] = None,
    min_score: Annotated[float, typer.Option("--min-score")] = 0.0,
    all_results: Annotated[bool, typer.Option("--all")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    files_out: Annotated[bool, typer.Option("--files")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.search(
        query,
        _filters(num, collection, path_prefix, after, before, min_width, min_height, min_score, all_results),
    )
    _emit_results(st.console, rows, json_out, files_out, debug=debug)


@app.command("vsearch")
def vsearch_cmd(
    ctx: typer.Context,
    query: str,
    num: Annotated[int, typer.Option("-n", "--num")] = 10,
    collection: Annotated[str | None, typer.Option("-c", "--collection")] = None,
    path_prefix: Annotated[str | None, typer.Option("--path-prefix")] = None,
    after: Annotated[str | None, typer.Option("--after")] = None,
    before: Annotated[str | None, typer.Option("--before")] = None,
    min_width: Annotated[int | None, typer.Option("--min-width")] = None,
    min_height: Annotated[int | None, typer.Option("--min-height")] = None,
    min_score: Annotated[float, typer.Option("--min-score")] = 0.75,
    all_results: Annotated[bool, typer.Option("--all")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    files_out: Annotated[bool, typer.Option("--files")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.vsearch(
        query,
        _filters(num, collection, path_prefix, after, before, min_width, min_height, min_score, all_results),
    )
    _emit_results(st.console, rows, json_out, files_out, debug=debug)


@app.command("query")
def query_cmd(
    ctx: typer.Context,
    query: str,
    num: Annotated[int, typer.Option("-n", "--num")] = 10,
    collection: Annotated[str | None, typer.Option("-c", "--collection")] = None,
    path_prefix: Annotated[str | None, typer.Option("--path-prefix")] = None,
    after: Annotated[str | None, typer.Option("--after")] = None,
    before: Annotated[str | None, typer.Option("--before")] = None,
    min_width: Annotated[int | None, typer.Option("--min-width")] = None,
    min_height: Annotated[int | None, typer.Option("--min-height")] = None,
    min_score: Annotated[float, typer.Option("--min-score")] = 0.75,
    all_results: Annotated[bool, typer.Option("--all")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    files_out: Annotated[bool, typer.Option("--files")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.query(
        query,
        _filters(num, collection, path_prefix, after, before, min_width, min_height, min_score, all_results),
    )
    _emit_results(st.console, rows, json_out, files_out, debug=debug)


@app.command("get")
def get_cmd(
    ctx: typer.Context,
    identifier: str,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    files_out: Annotated[bool, typer.Option("--files")] = False,
) -> None:
    st = _state(ctx)
    row = st.service.get(identifier)
    rows = [row] if row else []
    _emit_results(st.console, rows, json_out, files_out, debug=debug)


@app.command("multi-get")
def multi_get_cmd(
    ctx: typer.Context,
    identifiers: Annotated[list[str], typer.Argument()] = [],
    glob_pattern: Annotated[str | None, typer.Option("--glob")] = None,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    files_out: Annotated[bool, typer.Option("--files")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.multi_get(identifiers=identifiers, glob_pattern=glob_pattern)
    _emit_results(st.console, rows, json_out, files_out, debug=debug)


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    _emit_obj(st.console, st.service.status(), json_out)


@app.command("cleanup")
def cleanup_cmd(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    _emit_obj(st.console, st.service.cleanup(), json_out)


@app.command("repair")
def repair_cmd(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    st = _state(ctx)
    _emit_obj(st.console, st.service.repair(), json_out)


@app.command("similar")
def similar_cmd(
    ctx: typer.Context,
    identifier: str,
    num: Annotated[int, typer.Option("-n", "--num")] = 10,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    files_out: Annotated[bool, typer.Option("--files")] = False,
) -> None:
    st = _state(ctx)
    rows = st.service.similar(identifier, num=num)
    _emit_results(st.console, rows, json_out, files_out, debug=debug)


@app.command("mcp")
def mcp_cmd(
    ctx: typer.Context,
    action: Annotated[str, typer.Argument(help="start|stop")] = "start",
    http: Annotated[bool, typer.Option("--http", help="Use HTTP transport")] = False,
    daemon: Annotated[bool, typer.Option("--daemon", help="Run HTTP MCP server as daemon")] = False,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8181,
    serve_only: Annotated[bool, typer.Option("--serve-only", hidden=True)] = False,
) -> None:
    st = _state(ctx)
    pid_path = st.service.config.pid_path

    if action == "stop":
        if not pid_path.exists():
            typer.echo("not running")
            raise typer.Exit(0)
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        pid_path.unlink(missing_ok=True)
        typer.echo(f"stopped pid {pid}")
        raise typer.Exit(0)

    if daemon and not http:
        raise typer.BadParameter("--daemon currently requires --http")

    if daemon and not serve_only:
        cmd = [
            sys.executable,
            "-m",
            "qimg.cli",
            "--config",
            str(st.config_path),
            "mcp",
            "start",
            "--http",
            "--host",
            host,
            "--port",
            str(port),
            "--serve-only",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_path.write_text(str(proc.pid))
        typer.echo(f"started daemon pid={proc.pid} http://{host}:{port}")
        raise typer.Exit(0)

    if http:
        code = run_http_server(st.service, host=host, port=port)
        raise typer.Exit(code)

    code = run_stdio_server(st.service)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
