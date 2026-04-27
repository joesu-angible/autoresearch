from pathlib import Path

from research_loop.agents.author import PatchProposal
from research_loop.agents.worktree_authoring import author_diff_in_worktree, synthesize_diff_in_worktree


class EditingClient:
    name = "fake-editor"

    def __init__(self, edit_text: str) -> None:
        self.edit_text = edit_text
        self.calls = []

    def edit_files(self, system: str, user: str, *, workdir: Path, timeout: float | None = None) -> str:
        self.calls.append((system, user, workdir, timeout))
        target = workdir / "pkg" / "trainer.py"
        target.write_text(self.edit_text)
        return "edited"


class NoEditClient:
    name = "fake-no-editor"

    def __init__(self) -> None:
        self.calls = []

    def edit_files(self, system: str, user: str, *, workdir: Path, timeout: float | None = None) -> str:
        self.calls.append((system, user, workdir, timeout))
        return "No safe trainer change because prior failures look like orchestration noise."


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    trainer = repo / "pkg" / "trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("VALUE = 1\n")
    import subprocess

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "pkg/trainer.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def test_author_worktree_returns_git_generated_diff(tmp_path):
    repo = _init_repo(tmp_path)
    client = EditingClient("VALUE = 2\n")

    proposal = author_diff_in_worktree(
        client,
        repo_root=repo,
        trainer_path="pkg/trainer.py",
        trainer_source="VALUE = 1\n",
        critique_text="increase VALUE",
    )

    assert isinstance(proposal, PatchProposal)
    assert proposal.diff.startswith("diff --git a/pkg/trainer.py b/pkg/trainer.py")
    assert "-VALUE = 1" in proposal.diff
    assert "+VALUE = 2" in proposal.diff
    assert not (repo / "pkg" / "trainer.py").read_text() == "VALUE = 2\n"


def test_synthesizer_worktree_returns_empty_diff_when_agent_makes_no_changes(tmp_path):
    repo = _init_repo(tmp_path)
    client = EditingClient("VALUE = 1\n")

    proposal = synthesize_diff_in_worktree(
        client,
        repo_root=repo,
        trainer_path="pkg/trainer.py",
        patch_x="NO_PATCH",
        patch_y="diff --git a/pkg/trainer.py b/pkg/trainer.py\n",
    )

    assert proposal.diff == ""
    assert "No file changes" in proposal.rationale


def test_author_worktree_no_change_preserves_agent_rationale_and_pushes_safe_experiment(tmp_path):
    repo = _init_repo(tmp_path)
    client = NoEditClient()

    proposal = author_diff_in_worktree(
        client,
        repo_root=repo,
        trainer_path="pkg/trainer.py",
        trainer_source="VALUE = 1\n",
        critique_text="prior runs failed or timed out",
    )

    assert proposal.diff == ""
    assert "No safe trainer change" in proposal.rationale
    user_prompt = client.calls[0][1]
    assert "Prior failed/timeout outcomes are orchestration evidence" in user_prompt
    assert "generate at least one concrete safe trainer experiment" in user_prompt
