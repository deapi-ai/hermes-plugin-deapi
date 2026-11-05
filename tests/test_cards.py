"""Model-card math: sizes, sampling params and model choice."""

from __future__ import annotations

import pytest

from conftest import CATALOG, cards


def card(slug: str) -> dict:
    return next(m for m in CATALOG if m["slug"] == slug)


@pytest.mark.parametrize("slug, ratio, expected", [
    ("Flux1schnell", (3, 2), (768, 512)),            # default long side 768, 128 grid
    ("Flux1schnell", (2, 3), (512, 768)),
    ("Flux_2_Klein_4B_BF16", (3, 2), (1024, 672)),   # 16 grid snaps 682 down
    ("Flux_2_Klein_4B_BF16", (1, 1), (1024, 1024)),
    ("gpt-image-2", (3, 2), (1024, 640)),
])
def test_image_sizes_follow_the_card(slug, ratio, expected):
    assert cards.size_for_ratio(card(slug), *ratio) == expected


@pytest.mark.parametrize("ratio, short, expected", [
    ((16, 9), 720, (1280, 704)),   # 64 grid
    ((9, 16), 480, (512, 896)),    # grown to min side 512 first
    ((1, 1), 1080, (960, 960)),    # pixel budget 1032192, then the 64 grid
    ((21, 9), 720, (1344, 640)),   # max_ratio 2 caps 21:9, then the 1344 side limit
])
def test_video_sizes_follow_the_card(ratio, short, expected):
    w, h = cards.size_for_ratio(card("Ltx2_5_22B_Dist_INT8"), *ratio, short_side=short)
    limits = card("Ltx2_5_22B_Dist_INT8")["info"]["limits"]
    assert w * h <= limits["max_pixels"]
    assert w % 64 == 0 and h % 64 == 0
    assert (w, h) == expected


def test_sizes_never_exceed_side_limits():
    model = card("Ltx2_5_22B_Dist_INT8")
    for ratio in [(16, 9), (9, 16), (21, 9), (4, 3), (1, 1)]:
        for short in (480, 720, 1080):
            w, h = cards.size_for_ratio(model, *ratio, short_side=short)
            assert 512 <= w <= 1344 and 512 <= h <= 1344
            assert w * h <= 1032192


def test_sampling_fills_seed_and_card_steps():
    body = cards.apply_sampling(card("Flux1schnell"), {}, seed=7)
    assert body == {"seed": 7, "steps": 4}
    body = cards.apply_sampling(card("Ltx2_5_22B_Dist_INT8"), {})
    assert set(body) == {"seed"} and 0 <= body["seed"] < 2 ** 31
    assert cards.apply_sampling(card("Flux1schnell"), {}, seed=True)["seed"] is not True


def test_auto_pick_prefers_own_fleet():
    assert cards.auto_pick(CATALOG, "txt2img")["slug"] == "Flux_2_Klein_4B_BF16"
    assert cards.auto_pick(CATALOG, "img2img")["slug"] == "Flux_2_Klein_4B_BF16"
    assert cards.auto_pick(CATALOG, "txt2video")["slug"] == "Ltx2_5_22B_Dist_INT8"
    partner_only = [card("seedance-2.5")]
    assert cards.auto_pick(partner_only, "txt2video")["slug"] == "seedance-2.5"
    assert cards.auto_pick([card("Kokoro")], "txt2img") is None


def test_choose_explains_skipped_candidates():
    model, note = cards.choose(CATALOG, "txt2img", ["gpt-image-2-medium", None])
    assert model["slug"] == "Flux_2_Klein_4B_BF16"
    assert "not in this deAPI key's catalog" in note
    model, note = cards.choose(CATALOG, "img2img", ["Flux1schnell"])
    assert model["slug"] == "Flux_2_Klein_4B_BF16" and "does not serve img2img" in note
    model, note = cards.choose(CATALOG, "txt2img", ["gpt-image-2"])
    assert model["slug"] == "gpt-image-2" and note is None  # partner models are fine when chosen


def test_as_form_stringifies():
    assert cards.as_form({"seed": 1, "flag": True, "x": None, "p": "a"}) == {
        "seed": "1", "flag": "true", "p": "a"}
