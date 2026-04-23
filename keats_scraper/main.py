#!/usr/bin/env python3
"""
KEATS Student Handbook Scraper

A web scraper for extracting the Informatics Student Handbook from
King's College London's KEATS platform for use in a RAG chatbot.

Usage:
    python main.py login       # Authenticate with KEATS (manual 2FA)
    python main.py logout      # Clear saved session
    python main.py scrape      # Scrape handbook content (--resume supported)
    python main.py process     # Process and chunk documents
    python main.py all         # Run complete pipeline (scrape + process)
    python main.py status      # Show scraping progress
    python main.py validate    # Print content-quality report for scraped docs
    python main.py clear       # Delete all scraped data and checkpoints
"""

import json
import sys

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from keats_scraper.auth import SSOHandler
from keats_scraper.config import config
from keats_scraper.models import Document
from keats_scraper.processors import Chunker, ContentValidator, HTMLCleaner, TextNormalizer
from keats_scraper.scraper import CourseNavigator, PageScraper, PDFHandler, RateLimiter
from keats_scraper.storage import CheckpointManager, JSONLExporter
from keats_scraper.utils.exceptions import AuthenticationError, SessionExpiredError
from keats_scraper.utils.logging_config import get_logger, setup_logging

__version__ = "1.0.0"

console = Console()
logger = get_logger()


def setup_environment():
    """Initialize directories and logging."""
    config.ensure_directories()
    setup_logging(level=config.log_level, log_file=config.log_file)
    return get_logger()


def clean_and_validate_document(
    doc: Document,
    html_cleaner: HTMLCleaner,
    text_normalizer: TextNormalizer,
    content_validator: ContentValidator,
) -> Document | None:
    """Clean, normalise and validate a freshly scraped document.

    Returns None when the document has no content after cleaning, so
    the caller can drop it. Otherwise the document is returned with
    content populated. The ContentValidator runs as a soft filter:
    short, mostly-boilerplate, or metadata-light documents are
    reported via logger.warning (with URL, title and issue list) but
    are not dropped, so the coverage reconciliation step still sees
    them in the document index.

    Factored out of the scrape command's inner closure so the
    cleaning and validation path can be unit tested directly.
    """
    if doc.raw_html:
        cleaned = html_cleaner.clean(doc.raw_html)
        doc.content = text_normalizer.normalize(cleaned)
    else:
        doc.content = text_normalizer.normalize(doc.content)

    if not doc.content or not doc.content.strip():
        return None

    is_valid, issues = content_validator.validate_document(doc)
    if not is_valid or issues:
        logger.warning(
            "Low-quality document flagged by ContentValidator "
            "(url=%s, title=%r, issues=%s)",
            doc.metadata.source_url,
            doc.metadata.title,
            issues,
        )
    return doc


@click.group()
@click.version_option(version=__version__)
def cli():
    """KEATS Student Handbook Scraper for RAG Pipeline."""
    pass


@cli.command()
@click.option("--force", is_flag=True, help="Force new login even if session exists")
def login(force: bool):
    """Authenticate with KEATS (requires manual 2FA)."""
    setup_environment()

    console.print("\n[bold blue]KEATS Authentication[/bold blue]\n")

    sso = SSOHandler(config)

    if not force:
        # Check if we have a valid session
        cookies = sso.session_manager.load_cookies()
        if cookies:
            session = sso.session_manager.create_session_with_cookies(cookies)
            try:
                is_valid = sso.session_manager.validate_session(
                    session, config.auth.session_check_url
                )
            except SessionExpiredError:
                is_valid = False
            if is_valid:
                console.print("[green]Existing session is still valid![/green]")
                console.print("Use --force to re-authenticate anyway.")
                return

    try:
        # get_valid_session persists cookies via save_cookies as a
        # side effect of login_interactive; the returned Session object
        # is not needed here because later CLI invocations construct their
        # own sessions from the saved cookies.
        sso.get_valid_session(force_login=force)
        console.print("\n[bold green]Login successful![/bold green]")
        console.print("Session cookies have been saved for future use.")

    except Exception as e:
        console.print(f"\n[bold red]Login failed:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def logout():
    """Clear saved session."""
    setup_environment()

    sso = SSOHandler(config)
    sso.logout()
    console.print("[green]Session cleared.[/green]")


@cli.command()
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
def scrape(resume: bool):
    """Scrape handbook content from KEATS."""
    setup_environment()

    console.print("\n[bold blue]KEATS Handbook Scraper[/bold blue]\n")

    # Initialize components
    sso = SSOHandler(config)
    checkpoint = CheckpointManager(config.data_dir / "checkpoints")
    html_cleaner = HTMLCleaner()
    text_normalizer = TextNormalizer()
    content_validator = ContentValidator()

    # Check for valid session. Expected failures (no cookies, expired
    # session) surface the friendly "please re-login" message; anything
    # unexpected (missing ChromeDriver, network error, Selenium crash)
    # still emits the same user-facing text but writes a full traceback
    # to scraper.log via logger.exception so the root cause is
    # recoverable post-hoc.
    try:
        session = sso.get_valid_session()
    except (AuthenticationError, SessionExpiredError) as e:
        logger.warning("Expected auth failure in 'scrape' CLI: %s", e)
        console.print("[bold red]Authentication required.[/bold red]")
        console.print("Run: python main.py login")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected auth error in 'scrape' CLI")
        console.print("[bold red]Authentication required.[/bold red]")
        console.print("Run: python main.py login")
        sys.exit(1)

    rate_limiter = RateLimiter(config.rate_limit)
    navigator = CourseNavigator(session, config, rate_limiter)
    page_scraper = PageScraper(session, rate_limiter)
    pdf_handler = PDFHandler(session, rate_limiter, config)
    exporter = JSONLExporter(config.processed_dir)

    # Resume or start fresh
    progress_data = None
    if resume:
        progress_data = checkpoint.load()
        if progress_data:  # pragma: no branch - None-checkpoint resume path tested separately
            console.print("[yellow]Resuming from checkpoint...[/yellow]")
            console.print(f"Already processed: {len(progress_data.processed_urls)} URLs")

    # Discover resources
    console.print("\n[bold]Discovering handbook resources...[/bold]")
    try:
        resources = navigator.discover_resources()
    except Exception as e:
        console.print(f"[bold red]Failed to discover resources:[/bold red] {e}")
        console.print("Your session may have expired. Run: python main.py login")
        sys.exit(1)

    console.print(f"Found [bold]{len(resources)}[/bold] resources to process.\n")

    # Persist the discovery manifest. The list is held only in memory, so
    # a process crash or silent drop would otherwise be invisible after
    # the run. The coverage analyser reads this file to reconcile
    # discovered vs processed resources.
    manifest_path = config.data_dir / "checkpoints" / "discovered_resources.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for resource in resources:
            f.write(
                json.dumps(
                    {
                        "url": resource.url,
                        "title": resource.title,
                        "resource_type": resource.resource_type,
                        "section": resource.section,
                    }
                )
                + "\n"
            )
    logger.info(
        "Saved discovery manifest: %d resources at %s",
        len(resources),
        manifest_path,
    )

    if not progress_data:
        progress_data = checkpoint.start_new(len(resources))

    # Process resources
    documents: list[Document] = []

    def clean_document(doc: Document) -> Document | None:
        """Short alias for clean_and_validate_document so the scrape dispatch loop stays scannable."""
        return clean_and_validate_document(
            doc, html_cleaner, text_normalizer, content_validator
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Scraping...", total=len(resources))

        for resource in resources:
            # Skip if already processed
            if checkpoint.is_processed(resource.url):
                progress.advance(task)
                continue

            progress.update(task, description=f"Scraping: {resource.title[:40]}...")

            try:
                doc = None

                if resource.resource_type == "page":
                    doc = page_scraper.scrape_page(resource.url, resource.section)

                elif resource.resource_type == "forum":
                    # Forums can be scraped like pages (main discussion list)
                    doc = page_scraper.scrape_page(resource.url, resource.section)

                elif resource.resource_type == "glossary":
                    # Glossaries can be scraped like pages
                    doc = page_scraper.scrape_page(resource.url, resource.section)

                elif resource.resource_type in ("resource", "pdf"):
                    doc = pdf_handler.process_pdf(
                        resource.url, resource.title, resource.section
                    )

                elif resource.resource_type == "url":
                    # External link resources are out-of-scope for the
                    # scraped corpus (Threats to Validity, Evaluation Ch.).
                    # Log once and mark processed so they do not clog the
                    # failed_urls list; their titles are captured in the
                    # discovery manifest for reviewer context.
                    # increment_documents=False keeps documents_saved
                    # honest because no document is saved for these.
                    logger.info(
                        "Skipping external mod/url resource %r (%s)",
                        resource.title,
                        resource.url,
                    )
                    checkpoint.mark_processed(resource.url, increment_documents=False)
                    progress.advance(task)
                    continue

                elif resource.resource_type == "book":
                    # Books are containers: discover their chapters, scrape
                    # each chapter, then mark the book URL itself processed
                    # so a later --resume does not re-run the discovery
                    # network call. increment_documents=False keeps the
                    # documents_saved stat honest because the chapters
                    # are already counted individually below.
                    chapters = navigator.discover_book_chapters(resource.url, section=resource.section)
                    if not chapters:
                        logger.warning(f"No chapters discovered for book {resource.url}")
                        checkpoint.mark_failed(resource.url)
                    else:
                        for chapter in chapters:
                            if not checkpoint.is_processed(chapter.url):  # pragma: no branch - resume-skip path tested via fresh run
                                chapter_doc = page_scraper.scrape_page(
                                    chapter.url, resource.section
                                )
                                if chapter_doc:
                                    chapter_doc = clean_document(chapter_doc)
                                    if chapter_doc:
                                        documents.append(chapter_doc)
                                        checkpoint.mark_processed(chapter.url)
                                    else:
                                        logger.warning(f"Empty content after cleaning for {chapter.url}")
                                        checkpoint.mark_failed(chapter.url)
                                else:
                                    checkpoint.mark_failed(chapter.url)
                        checkpoint.mark_processed(resource.url, increment_documents=False)

                elif resource.resource_type == "folder":
                    # Folders are containers: same resume/stats rationale as
                    # the book branch above. The parent folder URL is marked
                    # processed once its files have been handled.
                    files = navigator.discover_folder_contents(resource.url, section=resource.section)
                    if not files:
                        logger.warning(f"No files discovered for folder {resource.url}")
                        checkpoint.mark_failed(resource.url)
                    else:
                        for file_info in files:
                            if not checkpoint.is_processed(file_info.url):  # pragma: no branch - resume-skip path tested separately
                                if file_info.resource_type == "pdf":
                                    file_doc = pdf_handler.process_pdf(
                                        file_info.url, file_info.title, resource.section
                                    )
                                else:
                                    file_doc = page_scraper.scrape_page(
                                        file_info.url, resource.section
                                    )
                                if file_doc:
                                    file_doc = clean_document(file_doc)
                                    if file_doc:
                                        documents.append(file_doc)
                                        checkpoint.mark_processed(file_info.url)
                                    else:
                                        logger.warning(f"Empty content after cleaning for {file_info.url}")
                                        checkpoint.mark_failed(file_info.url)
                                else:
                                    checkpoint.mark_failed(file_info.url)
                        checkpoint.mark_processed(resource.url, increment_documents=False)

                if doc:
                    doc = clean_document(doc)
                    if not doc:
                        logger.warning(f"Empty content after cleaning for {resource.url}")
                        checkpoint.mark_failed(resource.url)
                    else:
                        documents.append(doc)
                        checkpoint.mark_processed(resource.url)
                elif resource.resource_type not in ("book", "folder"):
                    checkpoint.mark_failed(resource.url)

            except Exception as e:
                logger.error(f"Failed to process {resource.url}: {e}")
                checkpoint.mark_failed(resource.url)

            progress.advance(task)

    # Save documents
    if documents:
        doc_path = exporter.export_documents(documents)
        console.print(f"\n[green]Saved {len(documents)} documents to {doc_path}[/green]")

    # Show stats
    stats = checkpoint.get_stats()
    console.print("\n[bold]Scraping Complete[/bold]")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Failed: {stats['failed']}")


@cli.command()
def process():
    """Process scraped documents into RAG-ready chunks."""
    setup_environment()

    console.print("\n[bold blue]Processing Documents into Chunks[/bold blue]\n")

    # Load documents
    doc_file = config.processed_dir / "documents.jsonl"
    if not doc_file.exists():
        console.print("[bold red]No documents found.[/bold red]")
        console.print("Run: python main.py scrape")
        sys.exit(1)

    documents = list(JSONLExporter.load_documents(doc_file))
    console.print(f"Loaded [bold]{len(documents)}[/bold] documents.")

    # Chunk documents
    chunker = Chunker(config.chunk)
    chunks = chunker.chunk_documents(documents)

    console.print(f"Created [bold]{len(chunks)}[/bold] chunks.")

    # Export chunks
    exporter = JSONLExporter(config.chunks_dir)
    chunk_path = exporter.export_chunks(chunks)
    embed_path = exporter.export_embedding_format(chunks)
    index_path = exporter.create_index(chunks)

    console.print("\n[green]Output files:[/green]")
    console.print(f"  Chunks: {chunk_path}")
    console.print(f"  Embeddings format: {embed_path}")
    console.print(f"  Index: {index_path}")


@cli.command(name="all")
@click.pass_context
def run_all(ctx):
    """Run complete pipeline: scrape and process."""
    setup_environment()

    console.print("\n[bold blue]Running Complete Pipeline[/bold blue]\n")

    console.print("[bold]Step 1: Scraping[/bold]")
    ctx.invoke(scrape)

    console.print("\n[bold]Step 2: Processing[/bold]")
    ctx.invoke(process)

    console.print("\n[bold green]Pipeline complete![/bold green]")


@cli.command()
def status():
    """Show current scraping progress."""
    setup_environment()

    checkpoint = CheckpointManager(config.data_dir / "checkpoints")
    stats = checkpoint.get_stats()

    console.print("\n[bold blue]Scraping Status[/bold blue]\n")

    if stats.get("status") == "no session":
        console.print("No scraping session found.")
        return

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Started", stats.get("started_at", "N/A"))
    table.add_row("Last Updated", stats.get("last_updated", "N/A"))
    table.add_row("Total Resources", str(stats.get("total_resources", 0)))
    table.add_row("Processed", str(stats.get("processed", 0)))
    table.add_row("Failed", str(stats.get("failed", 0)))
    table.add_row("Remaining", str(stats.get("remaining", 0)))
    table.add_row("Documents Saved", str(stats.get("documents_saved", 0)))

    console.print(table)


@cli.command()
def validate():
    """Validate scraped documents and print a content-quality report.

    Reads documents.jsonl from the processed directory and runs the
    ContentValidator over every record, emitting per-section
    coverage, word-count stats, and a table of documents flagged with
    content-quality issues (empty, too-short, boilerplate-heavy, or
    missing section metadata). Read-only -- it never modifies the data.
    """
    setup_environment()

    console.print("\n[bold blue]Content Quality Report[/bold blue]\n")

    documents_path = config.processed_dir / "documents.jsonl"
    if not documents_path.exists():
        console.print(
            f"[bold red]No documents found at {documents_path}.[/bold red]"
        )
        console.print("Run [cyan]python main.py scrape[/cyan] first.")
        sys.exit(1)

    try:
        documents = list(JSONLExporter.load_documents(documents_path))
    except Exception as e:
        console.print(f"[bold red]Failed to read {documents_path}:[/bold red] {e}")
        sys.exit(1)

    if not documents:
        console.print(
            f"[yellow]No documents to validate ({documents_path}).[/yellow]"
        )
        sys.exit(1)

    validator = ContentValidator()
    report = validator.generate_quality_report(documents)

    summary = Table(title="Overall")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Total documents", str(report["total_documents"]))
    summary.add_row("Valid", str(report["valid_documents"]))
    summary.add_row("Invalid", str(report["invalid_documents"]))
    summary.add_row("Empty", str(report["empty_documents"]))
    summary.add_row("Total words", str(report["total_words"]))
    summary.add_row("Avg words/doc", f"{report['avg_words_per_doc']:.1f}")
    summary.add_row("Min words", str(report["min_words"]))
    summary.add_row("Max words", str(report["max_words"]))
    summary.add_row("Sections covered", str(report["section_count"]))
    console.print(summary)

    sections = Table(title="Sections")
    sections.add_column("Section", style="cyan")
    for section in report["sections_covered"]:
        sections.add_row(section)
    console.print(sections)

    issues = report["documents_with_issues"]
    if issues:
        issues_table = Table(title=f"Documents with issues ({len(issues)})")
        issues_table.add_column("Document ID", style="cyan")
        issues_table.add_column("Title", style="white")
        issues_table.add_column("Issues", style="yellow")
        for doc_id, detail in issues.items():
            issues_table.add_row(
                doc_id,
                detail["title"] or "(untitled)",
                "; ".join(detail["issues"]),
            )
        console.print(issues_table)
    else:
        console.print("[green]No document-level issues reported.[/green]")


@cli.command()
def clear():
    """Clear all scraped data and checkpoints."""
    setup_environment()

    if click.confirm("This will delete all scraped data. Continue?"):
        import shutil

        # Clear data directories
        for directory in [config.raw_dir, config.processed_dir, config.chunks_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                directory.mkdir(parents=True)

        # Clear checkpoints
        checkpoint = CheckpointManager(config.data_dir / "checkpoints")
        checkpoint.clear()

        console.print("[green]All data cleared.[/green]")


if __name__ == "__main__":
    cli()
