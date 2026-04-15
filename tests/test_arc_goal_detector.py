"""
tests/test_arc_goal_detector.py — ARC goal detector unit tests.

Covers G1 empirical capture + G2 ls20 character/target detection + goal-distance
reward per rFP_arc_training_fix (2026-04-13).

Run:
    source test_env/bin/activate
    python -m pytest tests/test_arc_goal_detector.py -v -p no:anchorpy
"""
import json
import os

import numpy as np
import pytest

from titan_plugin.logic.arc.goal_detector import GoalDetector


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_grid(shape=(8, 8), fill=0):
    return np.full(shape, fill, dtype=np.int8)


@pytest.fixture
def gd(tmp_path, monkeypatch):
    """GoalDetector with isolated persistence path."""
    monkeypatch.chdir(tmp_path)
    return GoalDetector(persist_dir=str(tmp_path / "data" / "arc_agi_3"))


# ── G1: Empirical goal capture ───────────────────────────────────────

def test_no_goal_before_first_win(gd):
    assert gd.get_goal("ls20") is None
    assert gd.has_goal("ls20") is False


def test_capture_goal_on_win(gd):
    grid = _make_grid()
    grid[3:5, 3:5] = 5  # winning pattern
    gd.on_episode_end(game_id="ls20", final_state="WIN", final_grid=grid)
    assert gd.has_goal("ls20") is True
    stored = gd.get_goal("ls20")
    assert (stored == grid).all()


def test_no_capture_on_loss_or_timeout(gd):
    grid = _make_grid(fill=7)
    gd.on_episode_end(game_id="ls20", final_state="GAME_OVER", final_grid=grid)
    gd.on_episode_end(game_id="ls20", final_state="NOT_FINISHED", final_grid=grid)
    assert gd.has_goal("ls20") is False


def test_goal_persists_across_instances(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    persist = str(tmp_path / "data" / "arc_agi_3")
    gd1 = GoalDetector(persist_dir=persist)
    grid = _make_grid()
    grid[2, 2] = 9
    gd1.on_episode_end("ls20", "WIN", grid)
    # New instance reads persisted file
    gd2 = GoalDetector(persist_dir=persist)
    assert gd2.has_goal("ls20") is True
    assert (gd2.get_goal("ls20") == grid).all()


def test_per_game_goals_independent(gd):
    g_ls = _make_grid(); g_ls[0, 0] = 1
    g_ft = _make_grid(); g_ft[7, 7] = 2
    gd.on_episode_end("ls20", "WIN", g_ls)
    gd.on_episode_end("ft09", "WIN", g_ft)
    assert (gd.get_goal("ls20")[0, 0]) == 1
    assert (gd.get_goal("ft09")[7, 7]) == 2
    assert gd.get_goal("vc33") is None


# ── Similarity + goal distance ───────────────────────────────────────

def test_similarity_identical_grids(gd):
    a = _make_grid(fill=3)
    b = _make_grid(fill=3)
    assert gd.similarity(a, b) == 1.0


def test_similarity_all_different(gd):
    a = _make_grid(fill=0)
    b = _make_grid(fill=7)
    assert gd.similarity(a, b) == 0.0


def test_similarity_partial(gd):
    a = _make_grid(fill=0)
    b = _make_grid(fill=0)
    b[0, 0:4] = 1  # 4/64 cells differ
    sim = gd.similarity(a, b)
    assert abs(sim - (60 / 64)) < 1e-6


def test_goal_distance_reward_moving_closer(gd):
    goal = _make_grid(fill=0)
    goal[3, 3] = 5
    prev = _make_grid(fill=0)
    new = _make_grid(fill=0); new[3, 3] = 5  # matches goal
    reward = gd.goal_distance_delta(prev_grid=prev, new_grid=new, goal_grid=goal)
    assert reward > 0  # moved closer


def test_goal_distance_reward_moving_away(gd):
    goal = _make_grid(fill=0)
    goal[3, 3] = 5
    prev = _make_grid(fill=0); prev[3, 3] = 5  # matches goal
    new = _make_grid(fill=0); new[0, 0] = 5    # mismatch
    reward = gd.goal_distance_delta(prev_grid=prev, new_grid=new, goal_grid=goal)
    assert reward < 0


def test_goal_distance_reward_no_change(gd):
    goal = _make_grid(fill=0); goal[3, 3] = 5
    prev = _make_grid(fill=0); prev[7, 7] = 3
    new = _make_grid(fill=0); new[7, 7] = 3
    assert gd.goal_distance_delta(prev, new, goal) == 0.0


# ── G2: ls20 character detection ─────────────────────────────────────

def test_detect_ls20_character_simple_move(gd):
    """Character is a single cell that moves between frames."""
    prev = _make_grid(fill=0); prev[3, 3] = 1
    curr = _make_grid(fill=0); curr[3, 4] = 1
    pos = gd.detect_character(prev_grid=prev, curr_grid=curr)
    # Should be one of the two positions (character location detected)
    assert pos == (3, 4) or pos == (3, 3)


def test_detect_ls20_character_no_movement(gd):
    """No change between frames → no character detection."""
    prev = _make_grid(fill=0); prev[3, 3] = 1
    curr = _make_grid(fill=0); curr[3, 3] = 1
    assert gd.detect_character(prev, curr) is None


def test_detect_ls20_target_rare_cell(gd):
    """Target is the most distinctive (rare) non-background cell."""
    grid = _make_grid(fill=0)
    grid[0:4, 0:4] = 3  # common color — 16 cells
    grid[7, 7] = 9      # rare distinctive — 1 cell
    target = gd.detect_target(grid, background=0)
    assert target == (7, 7)


def test_detect_ls20_target_empty_grid_returns_none(gd):
    assert gd.detect_target(_make_grid(fill=0), background=0) is None


def test_character_target_distance_decreases(gd):
    """When character moves toward target, manhattan distance decreases."""
    # Character at (0,0), target at (4,4)
    prev = _make_grid(fill=0); prev[0, 0] = 1; prev[4, 4] = 9
    # Character moves to (1,1) — closer
    curr = _make_grid(fill=0); curr[1, 1] = 1; curr[4, 4] = 9
    delta = gd.character_target_reward(
        prev_grid=prev, curr_grid=curr, character_color=1, target=(4, 4),
    )
    assert delta > 0


def test_character_target_distance_increases(gd):
    prev = _make_grid(fill=0); prev[2, 2] = 1; prev[4, 4] = 9
    curr = _make_grid(fill=0); curr[0, 0] = 1; curr[4, 4] = 9
    delta = gd.character_target_reward(
        prev_grid=prev, curr_grid=curr, character_color=1, target=(4, 4),
    )
    assert delta < 0


# ── Persistence format ───────────────────────────────────────────────

def test_goal_grids_json_format(gd, tmp_path):
    grid = _make_grid(fill=0); grid[3, 3] = 5
    gd.on_episode_end("ls20", "WIN", grid)
    persisted = tmp_path / "data" / "arc_agi_3" / "goal_grids.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text())
    assert "ls20" in data
    assert "grid" in data["ls20"]
    assert "captured_at_utc" in data["ls20"]
    assert "shape" in data["ls20"]


# ── Fault tolerance ──────────────────────────────────────────────────

def test_corrupted_persistence_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    persist = tmp_path / "data" / "arc_agi_3"
    persist.mkdir(parents=True)
    (persist / "goal_grids.json").write_text("not valid json {{{")
    # Should not crash on init
    gd = GoalDetector(persist_dir=str(persist))
    assert gd.get_goal("ls20") is None
