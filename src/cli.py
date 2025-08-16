import sys
import click
from rich.console import Console
from .pipeline import generate_report

console = Console()

@click.command()
@click.option("--topic", "-t", required=True, help="Report topic/title prompt.")
@click.option("--words", "-w", required=True, type=int, help="Target word count.")
@click.option("--model", "-m", default="gpt-4o-mini", show_default=True)
def main(topic: str, words: int, model: str) -> None:
    """
    Generate a report to stdout. Redirect to a file if needed.
    """
    try:
        console.print(f"[bold]Generating report[/bold] (model={model}, words={words})…", style="cyan")
        report = generate_report(topic=topic, word_limit=words, model=model)
        sys.stdout.write(report.strip() + "\n")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
