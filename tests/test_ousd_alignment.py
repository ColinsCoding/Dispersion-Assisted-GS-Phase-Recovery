from dgs.ousd_alignment import CTA, components_to_ctas, stamp, print_alignment


def test_cta_registry_has_priority_1_and_2():
    priorities = {info["priority"] for info in CTA.values()}
    assert 1 in priorities
    assert 2 in priorities


def test_threat_reduction_is_priority_2_and_labeled_adjacent():
    info = CTA["Threat_Reduction_Adjacent"]
    assert info["priority"] == 2
    assert "DTRA" in info["description"]
    assert "NOT one of OUSD" in info["description"]


def test_components_to_ctas_single_component():
    ctas = components_to_ctas(["tsdft"])
    assert "FutureG" in ctas
    assert "Directed_Energy" in ctas


def test_components_to_ctas_dedupes():
    ctas = components_to_ctas(["tsdft", "gs_core"])
    assert len(ctas) == len(set(ctas))


def test_components_to_ctas_sorted_priority_then_name():
    ctas = components_to_ctas(list(CTA.keys()) and
                               [c for info in CTA.values() for c in info["repo_components"]])
    priorities = [CTA[c]["priority"] for c in ctas]
    assert priorities == sorted(priorities)


def test_components_to_ctas_unknown_component_ignored():
    ctas = components_to_ctas(["not_a_real_component"])
    assert ctas == []


def test_rogue_wave_maps_to_threat_reduction_and_sensing():
    ctas = components_to_ctas(["rogue_wave"])
    assert "Threat_Reduction_Adjacent" in ctas
    assert "Integrated_Sensing_and_Cyber" in ctas


def test_stamp_adds_ousd_key():
    stats = {"exit_code": 0}
    result = stamp(stats, components=["td_gs", "gs_fno"])
    assert "ousd" in result
    assert "aligned_ctas" in result["ousd"]
    assert "priority_1_ctas" in result["ousd"]


def test_stamp_classification_is_unclassified_dist_a():
    result = stamp({}, components=["tsdft"])
    assert result["ousd"]["classification"] == "UNCLASSIFIED // DISTRIBUTION A — Approved for Public Release"


def test_stamp_priority_1_subset_of_aligned():
    result = stamp({}, components=["tsdft", "rogue_wave", "lab_on_chip"])
    aligned = set(result["ousd"]["aligned_ctas"])
    pri1 = set(result["ousd"]["priority_1_ctas"])
    assert pri1.issubset(aligned)


def test_stamp_defaults_to_all_components():
    result = stamp({}, components=None)
    assert result["ousd"]["n_ctas"] == len(CTA)


def test_print_alignment_runs_without_error(capsys):
    print_alignment(components=["tsdft", "rogue_wave"])
    out = capsys.readouterr().out
    assert "OUSD(R&E)" in out
    assert "Total CTAs" in out
