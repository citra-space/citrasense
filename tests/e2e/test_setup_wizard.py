"""E2E: Walk through the setup wizard to configure a telescope sensor on DummyAdapter."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def test_fresh_start_shows_setup_wizard(app_page: Page):
    """A fresh CitraSense instance with no config shows the setup wizard modal."""
    wizard = app_page.locator("#setupWizard")
    expect(wizard).to_be_visible(timeout=5000)
    expect(wizard.locator(".modal-title")).to_have_text("Welcome to CitraSense Setup")


def test_wizard_navigates_to_config(app_page: Page):
    """Clicking 'Configure Now' in the wizard navigates to the config page."""
    wizard = app_page.locator("#setupWizard")
    expect(wizard).to_be_visible(timeout=5000)

    wizard.locator("button:has-text('Configure Now')").click()

    # SPA uses history.pushState — check pathname, not hash
    app_page.wait_for_function("() => location.pathname.includes('config')")
    expect(wizard).to_be_hidden()


def test_enable_dummy_api(app_page: Page):
    """Enable the Dummy API via Advanced settings."""
    # Navigate to config via wizard
    app_page.locator("#setupWizard").wait_for(state="visible", timeout=5000)
    app_page.locator("#setupWizard button:has-text('Configure Now')").click()
    app_page.wait_for_function("() => location.pathname.includes('config')")

    # Go to Advanced tab (use the nav link, not the in-page alert link)
    app_page.locator(".nav-link:has-text('Advanced')").click()

    # Enable dummy API checkbox
    dummy_checkbox = app_page.locator("#use_dummy_api")
    dummy_checkbox.scroll_into_view_if_needed()
    dummy_checkbox.check()
    expect(dummy_checkbox).to_be_checked()


def test_add_telescope_sensor_with_dummy_adapter(app_page: Page):
    """Full flow: enable dummy API, add a telescope sensor with dummy adapter, save, and connect."""
    # Dismiss wizard → config
    app_page.locator("#setupWizard").wait_for(state="visible", timeout=5000)
    app_page.locator("#setupWizard button:has-text('Configure Now')").click()
    app_page.wait_for_function("() => location.pathname.includes('config')")

    # Add a sensor first (required fields on hardware tab block save when no sensors exist)
    app_page.locator("a:has-text('Add Sensor')").click()
    app_page.locator("input[placeholder*='Sensor ID']").fill("scope-1")
    type_select = app_page.locator("select").filter(has=app_page.locator("option[value='telescope']")).first
    type_select.select_option("telescope")
    adapter_select = app_page.locator("select").filter(has=app_page.locator("option:has-text('-- adapter --')")).first
    adapter_select.select_option("dummy")
    app_page.locator("button:has-text('Add')").first.click()
    app_page.locator("a:has-text('scope-1')").wait_for(state="visible", timeout=5000)

    # Navigate to the sensor's hardware tab and fill the Citra Sensor ID
    app_page.locator("a:has-text('scope-1')").click()
    app_page.locator("#citraSensorId").wait_for(state="visible", timeout=5000)
    app_page.locator("#citraSensorId").fill("test-sensor-uuid")

    # Enable dummy API in Advanced
    app_page.locator(".nav-link:has-text('Advanced')").click()
    app_page.locator("#use_dummy_api").check()

    # Fill API token
    app_page.locator(".nav-link:has-text('API')").click()
    app_page.locator("#personal_access_token").fill("test-token")

    # Save everything at once (reload_configuration may take time as the
    # DummyApiClient fetches satellite TLEs from the network on first init)
    app_page.locator("button:has-text('Save Configuration')").click()
    app_page.locator(".toast:has-text('Configuration')").wait_for(state="visible", timeout=30000)

    # Navigate to monitoring page
    app_page.locator("a:has-text('Monitoring')").first.click()
    app_page.wait_for_function("() => location.pathname === '/' || location.pathname.includes('monitoring')")

    # The sensor should appear in the monitoring section header
    app_page.locator("#monitoringSection").get_by_text("scope-1").first.wait_for(state="visible", timeout=10000)
