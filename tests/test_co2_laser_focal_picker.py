"""Tests for dgs/co2_laser_focal_picker.py"""
import pytest

from dgs.co2_laser_focal_picker import (
    focused_spot_and_depth_of_focus, recommend_focal_length,
    default_raw_beam_radius_mm, STANDARD_FOCAL_LENGTHS_INCH,
)


class TestFocusedSpotAndDepthOfFocus:
    def test_shorter_focal_length_gives_smaller_spot(self):
        spot_short, _ = focused_spot_and_depth_of_focus(1.5, raw_beam_radius_mm=4.0)
        spot_long, _ = focused_spot_and_depth_of_focus(2.5, raw_beam_radius_mm=4.0)
        assert spot_short < spot_long

    def test_shorter_focal_length_gives_shorter_depth_of_focus(self):
        _, dof_short = focused_spot_and_depth_of_focus(1.5, raw_beam_radius_mm=4.0)
        _, dof_long = focused_spot_and_depth_of_focus(2.5, raw_beam_radius_mm=4.0)
        assert dof_short < dof_long

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            focused_spot_and_depth_of_focus(0, raw_beam_radius_mm=4.0)
        with pytest.raises(ValueError):
            focused_spot_and_depth_of_focus(2.0, raw_beam_radius_mm=0)


class TestDefaultBeamRadius:
    def test_known_power_lookup(self):
        assert default_raw_beam_radius_mm(60) == 3.5
        assert default_raw_beam_radius_mm(100) == 4.5

    def test_nearest_power_lookup(self):
        # 65W isn't a table entry -- should snap to the nearest (60W)
        assert default_raw_beam_radius_mm(65) == default_raw_beam_radius_mm(60)

    def test_rejects_nonpositive_power(self):
        with pytest.raises(ValueError):
            default_raw_beam_radius_mm(0)


class TestRecommendFocalLength:
    def test_engrave_always_picks_shortest(self):
        r = recommend_focal_length(use_case="engrave", power_W=60.0)
        assert r["recommended_focal_length_inch"] == min(STANDARD_FOCAL_LENGTHS_INCH)

    def test_engrave_ignores_material_thickness(self):
        r1 = recommend_focal_length(use_case="engrave", power_W=60.0)
        r2 = recommend_focal_length(use_case="engrave", power_W=60.0, material_thickness_mm=50.0)
        assert r1["recommended_focal_length_inch"] == r2["recommended_focal_length_inch"]

    def test_cut_requires_material_thickness(self):
        with pytest.raises(ValueError):
            recommend_focal_length(use_case="cut", power_W=60.0)

    def test_cut_rejects_bad_use_case(self):
        with pytest.raises(ValueError):
            recommend_focal_length(use_case="ablate", material_thickness_mm=3.0)

    def test_thin_material_gets_short_focal_length(self):
        r = recommend_focal_length(use_case="cut", material_thickness_mm=0.5, power_W=60.0)
        assert r["recommended_focal_length_inch"] == min(STANDARD_FOCAL_LENGTHS_INCH)

    def test_thicker_material_never_gets_a_shorter_recommendation(self):
        # monotonicity: recommended focal length must be non-decreasing as material gets thicker
        thicknesses = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
        recs = [recommend_focal_length(use_case="cut", material_thickness_mm=t, power_W=60.0)
                ["recommended_focal_length_inch"] for t in thicknesses]
        assert all(recs[i] <= recs[i + 1] for i in range(len(recs) - 1))

    def test_recommended_depth_of_focus_covers_material_when_possible(self):
        r = recommend_focal_length(use_case="cut", material_thickness_mm=3.0, power_W=60.0)
        # 3mm is coverable by some standard lens at 60W (verified in the module's own demo) --
        # the recommendation must actually satisfy the physical requirement it claims to
        assert r["predicted_depth_of_focus_mm"] >= 3.0

    def test_impossible_thickness_falls_back_to_longest_with_honest_reason(self):
        r = recommend_focal_length(use_case="cut", material_thickness_mm=1000.0, power_W=60.0)
        assert r["recommended_focal_length_inch"] == max(STANDARD_FOCAL_LENGTHS_INCH)
        assert "NO standard lens" in r["reason"]

    def test_full_comparison_table_has_one_row_per_focal_length(self):
        r = recommend_focal_length(use_case="engrave", power_W=60.0)
        assert len(r["full_comparison_table"]) == len(STANDARD_FOCAL_LENGTHS_INCH)
        assert [row["focal_length_inch"] for row in r["full_comparison_table"]] == sorted(STANDARD_FOCAL_LENGTHS_INCH)

    def test_manual_beam_radius_overrides_power_default(self):
        r_default = recommend_focal_length(use_case="engrave", power_W=60.0)
        r_manual = recommend_focal_length(use_case="engrave", power_W=60.0, raw_beam_radius_mm=10.0)
        assert r_default["raw_beam_radius_mm"] != r_manual["raw_beam_radius_mm"]
        assert r_manual["raw_beam_radius_mm"] == 10.0
