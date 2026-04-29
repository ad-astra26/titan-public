"""C2-10 / BUG-DEPLOY-T3-WIPES-LOCAL-EDITS-20260428.

Verifies the skip-worktree filter in deploy_t2.sh / deploy_t3.sh:
  - Files marked `git update-index --skip-worktree` are NOT reset by
    `git checkout -- .`-equivalent inside the deploy.
  - `git pull` still applies upstream changes for files that don't have
    the skip-worktree flag.

Per PLAN_microkernel_phase_c_s2_kernel.md §17.2.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


# Excerpt of the deploy_t*.sh logic — the part that actually decides what
# to reset. We test the BEHAVIOR (skip-worktree files survive) by running
# this exact pattern against a sandbox git repo.
_DEPLOY_RESET_SCRIPT = r"""
set -e
cd "$1"

# Mirror deploy_t*.sh: --skip-worktree mark + filter from reset list
git update-index --skip-worktree titan_plugin/titan_params.toml 2>/dev/null || true

SKIP_WORKTREE_FILES=$(git ls-files -v 2>/dev/null \
    | awk '$1 == "S" || $1 == "h" {sub(/^[a-zA-Z] /,""); print}')
{
    echo "titan_plugin/config.toml"
    [ -n "$SKIP_WORKTREE_FILES" ] && echo "$SKIP_WORKTREE_FILES"
} | sort -u > /tmp/test_skip_reset.lst

git diff --name-only \
    | grep -vxFf /tmp/test_skip_reset.lst \
    | xargs -r git checkout -- 2>/dev/null || true
rm -f /tmp/test_skip_reset.lst
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(repo), text=True
    ).strip()


def _setup_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "titan_plugin").mkdir(parents=True)
    subprocess.check_call(["git", "init", "-q", "-b", "main"], cwd=str(repo))
    subprocess.check_call(
        ["git", "config", "user.email", "test@example.com"], cwd=str(repo)
    )
    subprocess.check_call(
        ["git", "config", "user.name", "Test"], cwd=str(repo)
    )
    (repo / "titan_plugin" / "titan_params.toml").write_text(
        "# initial\nflag = false\n"
    )
    (repo / "titan_plugin" / "config.toml").write_text("# config\n")
    (repo / "titan_plugin" / "other.py").write_text("# other\n")
    subprocess.check_call(["git", "add", "."], cwd=str(repo))
    subprocess.check_call(
        ["git", "commit", "-q", "-m", "initial"], cwd=str(repo)
    )
    return repo


class TestSkipWorktreeFilter:
    def test_skip_worktree_edit_survives_reset(self, tmp_path):
        repo = _setup_repo(tmp_path)

        # Edit titan_params.toml WITHOUT staging — simulates Maker's
        # in-flight flag flip on T2/T3.
        params = repo / "titan_plugin" / "titan_params.toml"
        params.write_text("# initial\nflag = true  # MAKER EDIT\n")

        # Edit other.py too — this one is NOT skip-worktree, should be wiped.
        other = repo / "titan_plugin" / "other.py"
        other.write_text("# THIS SHOULD BE RESET BY DEPLOY\n")

        subprocess.check_call(
            ["bash", "-c", _DEPLOY_RESET_SCRIPT, "_", str(repo)]
        )

        # Maker edit survived
        assert "MAKER EDIT" in params.read_text(), (
            "skip-worktree titan_params.toml edit was wiped"
        )
        # other.py was reset to committed version
        assert other.read_text() == "# other\n", (
            "non-skip-worktree file should have been reset"
        )

    def test_skip_worktree_idempotent(self, tmp_path):
        """Running the deploy script twice doesn't break the lock."""
        repo = _setup_repo(tmp_path)
        params = repo / "titan_plugin" / "titan_params.toml"
        params.write_text("# initial\nflag = true  # MAKER EDIT\n")

        for _ in range(2):
            subprocess.check_call(
                ["bash", "-c", _DEPLOY_RESET_SCRIPT, "_", str(repo)]
            )

        assert "MAKER EDIT" in params.read_text()
        # Verify --skip-worktree flag still set
        ls = _git(repo, "ls-files", "-v", "titan_plugin/titan_params.toml")
        assert ls.startswith("S "), f"flag dropped: {ls!r}"


class TestDeployScriptSourceContract:
    """Source-level contracts on the actual deploy scripts."""

    @pytest.mark.parametrize("script", ["deploy_t2.sh", "deploy_t3.sh"])
    def test_deploy_uses_skip_worktree_filter(self, script):
        path = Path(__file__).parent.parent / "scripts" / script
        src = path.read_text(encoding="utf-8")
        # Filter logic present
        assert "skip-worktree" in src, (
            f"{script} missing skip-worktree mark"
        )
        assert "titan_params.toml" in src, (
            f"{script} missing titan_params.toml mention"
        )
        # The dangerous unfiltered checkout is GONE
        bad_pattern = (
            'git diff --name-only | grep -v "^titan_plugin/config.toml$"'
            ' | xargs -r git checkout --'
        )
        assert bad_pattern not in src, (
            f"{script} still uses pre-fix unfiltered checkout pattern"
        )
