from pathlib import Path

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
@click.option(
    "--base-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Root all state (config, data, logs, cache) under this directory.",
)
def cli(web_port, base_dir):
    """CitraSense daemon - configure via web UI at http://localhost:24872"""
    settings = CitraSenseSettings.load(web_port=web_port, base_dir=base_dir)
    daemon = CitraSenseDaemon(settings)
    daemon.run()


if __name__ == "__main__":
    cli()
