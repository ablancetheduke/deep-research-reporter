import sys
import click
from rich.console import Console

from . import pipeline

console = Console()


def _load_topic(topic: str | None, topic_file: str | None) -> str:
    if topic and topic.strip():
        return topic.strip()
    if topic_file:
        try:
            with open(topic_file, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
        except Exception as e:
            console.print(f"[red]Failed to read topic file:[/red] {e}")
            sys.exit(1)
    console.print("[red]Please provide --topic or --topic-file.[/red]")
    sys.exit(1)


@click.command()
@click.option("--topic", "-t", help="Report topic/title prompt.")
@click.option("--topic-file", "-f", help="Read topic from a text file.")
@click.option("--words", "-w", required=True, type=int, help="Target word count.")
@click.option("--provider", "-p",
              type=click.Choice(["openai", "gemini", "deepseek", "chatglm", "mcp"], case_sensitive=False),
              default="openai", show_default=True,
              help="LLM provider.")
@click.option("--model", "-m", default="gpt-4o-mini", show_default=True,
              help="Model name for the chosen provider.")
@click.option("--out", "-o", help="Write report to file (Markdown). If omitted, print to stdout.")
def main(topic: str | None,
         topic_file: str | None,
         words: int,
         provider: str,
         model: str,
         out: str | None) -> None:
    """
    Generate a structured analysis report.

    Example:
      python -m drr.cli -t "AI in healthcare" -w 1000 -p gemini -m gemini-1.5-flash -o report.md
    """
    topic_text = _load_topic(topic, topic_file)

    console.print(
        f"[bold]Generating report[/bold] "
        f"(provider={provider}, model={model}, words={words})…",
        style="cyan"
    )

    try:
        # Backward-compatible invocation:
        if hasattr(pipeline, "generate_report_v2"):
            report = pipeline.generate_report_v2(
                topic=topic_text,
                word_limit=words,
                provider=provider,
                model=model,
            )
        else:
            # Legacy pipeline without provider support.
            if provider.lower() != "openai":
                console.print(
                    "[yellow]Note:[/] current pipeline doesn't accept --provider yet; "
                    "falling back to OpenAI-compatible path.",
                )
            report = pipeline.generate_report(
                topic=topic_text,
                word_limit=words,
                model=model,
            )

        if out:
            with open(out, "w", encoding="utf-8") as f:
                f.write(report.strip() + "\n")
            console.print(f"[green]Saved to[/green] {out}")
        else:
            sys.stdout.write(report.strip() + "\n")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
