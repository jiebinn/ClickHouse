import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ci.jobs import cloud_api_docs_nightly


def test_push_uses_app_token_without_verbose_logging(monkeypatch):
    checks = []

    def fake_check(command, verbose=False, **_kwargs):
        checks.append((command, verbose))
        return True

    monkeypatch.setattr(cloud_api_docs_nightly.Shell, "check", fake_check)
    monkeypatch.setattr(
        cloud_api_docs_nightly.Shell,
        "get_output",
        lambda *_args, **_kwargs: "123",
    )

    assert cloud_api_docs_nightly.open_or_refresh_pr()
    assert len(checks) == 2

    prepare, push = checks
    assert prepare[1] is True
    assert "git commit" in prepare[0]
    assert "git push" not in prepare[0]

    assert push[1] is False
    assert 'token="$(gh auth token)"' in push[0]
    assert "http.https://github.com/.extraheader=" in push[0]
    assert "ClickHouse/ClickHouse.git" in push[0]
    assert "robot/cloud-api-docs:refs/heads/robot/cloud-api-docs" in push[0]
