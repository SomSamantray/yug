# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""The ``yug-check`` pre-flight: it must look at the drive the weights land on."""

from __future__ import annotations

import shutil
from collections import namedtuple
from pathlib import Path

import pytest
from huggingface_hub import constants

from yug import __main__ as check


@pytest.fixture
def disk_usage_calls(monkeypatch) -> list[str]:
    """Record every path ``shutil.disk_usage`` is asked about."""
    calls: list[str] = []
    real = shutil.disk_usage

    def spy(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(shutil, "disk_usage", spy)
    return calls


def test_free_disk_is_measured_where_the_weights_are_cached(
    tmp_path, monkeypatch, disk_usage_calls
):
    """Checking the working directory instead would report the wrong drive."""
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    check.main(["--quiet"])

    assert disk_usage_calls == [str(cache)]


def test_a_cache_that_does_not_exist_yet_falls_back_to_its_nearest_parent(
    tmp_path, monkeypatch, disk_usage_calls
):
    """On a first run the cache folder is absent; ``disk_usage`` would raise."""
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "a" / "b" / "hub"))

    check.main(["--quiet"])

    assert disk_usage_calls == [str(tmp_path)]


def test_nearest_existing_parent_stops_at_the_filesystem_root():
    missing = Path("/definitely/not/here/at/all")
    assert check._nearest_existing(missing) == Path(missing.anchor)


def test_low_disk_blocks_and_names_the_cache_folder(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**9, 10**9, 10**6))

    code = check.main([])

    assert code == 1
    assert f"in {cache}" in capsys.readouterr().out
