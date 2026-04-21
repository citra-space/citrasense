import click

from citrasense.citrasense_daemon import CitraSenseDaemon
from citrasense.constants import DEFAULT_WEB_PORT
from citrasense.settings.citrasense_settings import CitraSenseSettings
from citrasense.version import format_version_cli, get_version_info


def _version_string() -> str:
    return format_version_cli(get_version_info())


@click.command()
@click.version_option(version=_version_string(), prog_name="citrasense")
@click.option(
    "--web-port",
    default=DEFAULT_WEB_PORT,
    type=int,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
)
def cli(web_port):
    """CitraSense daemon - configure via web UI at http://localhost:24872"""
    settings = CitraSenseSettings.load(web_port=web_port)
    daemon = CitraSenseDaemon(settings)
    daemon.run()


if __name__ == "__main__":
    cli()
