# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the machine profile isolation of the test session (tests/conftest.py).

The tests must not read the tuner profile of the machine. The conftest
points ``VENTORUM_CONFIG_DIR`` to an empty folder for the session; a test
of the profile behaviour opts in with its own folder or with the
``machine_profile`` fixture.
"""

import os
import subprocess
import sys

from ventorum.hardware import profile


def test_session_sees_no_machine_profile():
    folder = os.environ.get("VENTORUM_CONFIG_DIR")
    assert folder, "the conftest did not set VENTORUM_CONFIG_DIR"
    assert profile.config_dir() == profile.profile_path().parent
    assert str(profile.config_dir()) == folder
    assert not profile.profile_path().exists()
    profile.reset_profile_cache()
    assert profile.active_profile() is None


def test_subprocess_sees_no_machine_profile():
    code = "from ventorum.hardware import profile; print(profile.profile_path().exists())"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_own_folder_opts_in(tmp_path, monkeypatch):
    """A test that sets its own folder reads the profile that it writes."""
    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    profile.reset_profile_cache()
    try:
        assert profile.profile_path() == tmp_path / profile.PROFILE_FILENAME
    finally:
        profile.reset_profile_cache()


def test_machine_profile_fixture_restores_the_real_folder(machine_profile):
    assert machine_profile == profile.profile_path()
    session_folder = os.environ.get("VENTORUM_CONFIG_DIR")
    assert session_folder is None or "ventorum_no_machine_profile_" not in session_folder
