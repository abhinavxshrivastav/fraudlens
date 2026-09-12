"""Tests for configuration path resolution.

The bug these pin was silent. Paths used to be derived from ``__file__``, which
works from a source checkout and nowhere else: under a non-editable
``pip install .`` the package lives in site-packages, so the service found no
rule file and started with an empty rule set. That still scores transactions --
it just stops enforcing every deterministic block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fraudlens.config import Settings


def test_working_directory_layout_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The container case: WORKDIR holds config/ and must win over the source tree.
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "rules.yaml").write_text("version: test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = Settings(_env_file=None).resolved_rules_path

    assert resolved.resolve() == (tmp_path / "config" / "rules.yaml").resolve()


def test_falls_back_to_the_source_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Tests and scripts run from a subdirectory still find the real rule set.
    monkeypatch.chdir(tmp_path)

    resolved = Settings(_env_file=None).resolved_rules_path

    assert resolved.is_file()
    assert resolved.name == "rules.yaml"


def test_absolute_path_is_used_as_given(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.yaml"

    assert Settings(_env_file=None, rules_path=target).resolved_rules_path == target


def test_unresolvable_path_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # So the "not found" log names the path that was configured, not a guess.
    monkeypatch.chdir(tmp_path)
    missing = Path("no/such/console")

    assert Settings(_env_file=None, console_dir=missing).resolved_console_dir == missing
