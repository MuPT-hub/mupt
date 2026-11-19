"""Unit tests for chain-size, sequence, and tacticity models"""

import pytest
import numpy as np

from mupt.polymers.models import (
    SequenceAlternating,
    TacticityBernoullian,
    SequenceBlock,
    TacticityEnantiomorphicSite,
    SequenceExplicit,
    TacticityMarkov,
    SizeFlory,
    SizeMonodisperse,
    SequenceRandom,
    SizeSchulzZimm,
    SequenceMayoLewis,
    SizeUniform,
    TacticityConstant,
    atactic,
    coerce_size,
    syndiospecific_site,
    configs_to_diads,
    gaps_between,
    isotactic,
    meso_fraction,
    syndiotactic,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# Chain sizes
def dp_stats(sizes: list[int]) -> tuple[float, float]:
    sizes = np.asarray(sizes, dtype=float)
    return sizes.mean(), (sizes**2).sum() / sizes.sum() / sizes.mean()


def test_monodisperse_and_coercion(rng) -> None:
    assert SizeMonodisperse(5).sample(3, rng) == [5, 5, 5]
    assert isinstance(coerce_size(7), SizeMonodisperse)
    with pytest.raises(TypeError):
        coerce_size(None)


def test_uniform_size_is_inclusive(rng) -> None:
    assert set(SizeUniform(2, 4).sample(2000, rng)) == {2, 3, 4}


def test_schulz_zimm_moments(rng) -> None:
    mean, pdi = dp_stats(SizeSchulzZimm(200, 1.5).sample(50_000, rng))
    assert mean == pytest.approx(200, rel=0.02)
    assert pdi == pytest.approx(1.5, rel=0.02)
    with pytest.raises(ValueError):
        SizeSchulzZimm(200, 1.0)


def test_flory_moments(rng) -> None:
    sizes = SizeFlory(50).sample(50_000, rng)
    mean, pdi = dp_stats(sizes)
    assert min(sizes) >= 1
    assert mean == pytest.approx(50, rel=0.03)
    assert pdi == pytest.approx(2 - 1 / 50, rel=0.03)


def test_samples_are_native_ints(rng) -> None:
    assert all(type(size) is int for size in SizeSchulzZimm(3, 2.0).sample(100, rng))


# Sequences
def test_random_copolymer_composition(rng) -> None:
    seq = SequenceRandom({"A": 3, "B": 1}).sample(20_000, rng)
    assert seq.count("A") / len(seq) == pytest.approx(0.75, abs=0.02)
    assert all(type(label) is str for label in seq)


def test_terminal_model_ideal_reduces_to_random() -> None:
    model = SequenceMayoLewis({"A": 1, "B": 3})
    np.testing.assert_allclose(model.transition, [[0.25, 0.75], [0.25, 0.75]])
    np.testing.assert_allclose(model.stationary, [0.25, 0.75])


def test_terminal_model_stationary_and_sampling(rng) -> None:
    model = SequenceMayoLewis({"A": 1, "B": 1}, reactivity={("A", "B"): 0.5, ("B", "A"): 2.0})
    np.testing.assert_allclose(model.transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.stationary @ model.transition, model.stationary)

    seq = model.sample(50_000, rng)
    assert seq.count("A") / len(seq) == pytest.approx(model.stationary[0], abs=0.02)


@pytest.mark.parametrize("reactivity", [{("A", "B"): 0.0}, {("A", "C"): 1.0}, {("A", "A"): 2.0}])
def test_terminal_model_rejects_bad_reactivity(reactivity) -> None:
    with pytest.raises(ValueError):
        SequenceMayoLewis({"A": 1, "B": 1}, reactivity=reactivity)


def test_alternating_block_explicit(rng) -> None:
    assert SequenceAlternating(["A", "B"]).sample(5, rng) == ["A", "B", "A", "B", "A"]
    assert SequenceBlock([("A", 1), ("B", 2)]).sample(10, rng) == ["A"] * 3 + ["B"] * 7
    assert SequenceExplicit(["A", "B"]).sample(2, rng) == ["A", "B"]
    with pytest.raises(ValueError):
        SequenceExplicit(["A", "B"]).sample(3, rng)


# Tacticity
@pytest.mark.parametrize("model,expected", [(isotactic(), 1.0), (syndiotactic(), 0.0)])
def test_tacticity_presets(model, expected, rng) -> None:
    assert meso_fraction(model.sample(50, rng)) == expected


@pytest.mark.parametrize("p_m", [0.2, 0.5, 0.8])
def test_bernoullian_meso_fraction(p_m, rng) -> None:
    assert meso_fraction(TacticityBernoullian(p_m).sample(20_000, rng)) == pytest.approx(p_m, abs=0.02)


def test_first_order_markov_conditional_diads(rng) -> None:
    diads = configs_to_diads(TacticityMarkov(0.9, 0.3).sample(50_000, rng))
    after_m = [b for a, b in zip(diads[:-1], diads[1:]) if a == "m"]
    after_r = [b for a, b in zip(diads[:-1], diads[1:]) if a == "r"]
    assert after_m.count("m") / len(after_m) == pytest.approx(0.9, abs=0.02)
    assert after_r.count("m") / len(after_r) == pytest.approx(0.3, abs=0.02)


def test_enantiomorphic_site_meso_fraction(rng) -> None:
    sigma = 0.8
    configs = TacticityEnantiomorphicSite(sigma).sample(20_000, rng)
    assert meso_fraction(configs) == pytest.approx(sigma**2 + (1 - sigma) ** 2, abs=0.02)


def test_tacticity_handles_empty(rng) -> None:
    assert atactic().sample(0, rng) == []
    assert atactic().sample([], rng) == []


# Tacticity across non-tactic units (e.g. styrene units separated by ethylene)
def test_gaps_between() -> None:
    # E S E E S S  ->  styrenes preceded by 1, 2, and 0 ethylenes
    assert gaps_between([False, True, False, False, True, True]) == [1, 2, 0]
    assert gaps_between([True, True, True]) == [0, 0, 0]
    assert gaps_between([False, False]) == []


ALL_MODELS = [TacticityBernoullian(0.7), TacticityMarkov(0.8, 0.3), TacticityEnantiomorphicSite(0.6), TacticityConstant(), TacticityConstant(-1)]


@pytest.mark.parametrize("model", ALL_MODELS)
def test_an_int_means_adjacent_units(model) -> None:
    from_int = model.sample(30, np.random.default_rng(5))
    from_gaps = model.sample([0] * 30, np.random.default_rng(5))
    assert from_int == from_gaps


@pytest.mark.parametrize("model", ALL_MODELS)
def test_one_draw_per_tactic_unit_whatever_the_gaps(model) -> None:
    gapped, adjacent = np.random.default_rng(2), np.random.default_rng(2)
    model.sample([0, 3, 0, 1, 0], gapped)
    model.sample(5, adjacent)
    assert gapped.bit_generator.state == adjacent.bit_generator.state


def test_reset_makes_each_run_its_own_stereo_block(rng) -> None:
    # two runs of three adjacent units, separated by a gap
    gaps = [0, 0, 0, 4, 0, 0]
    for _ in range(20):
        configs = isotactic().sample(gaps, rng)
        assert configs[0] == configs[1] == configs[2]
        assert configs[3] == configs[4] == configs[5]


@pytest.mark.parametrize("model", [isotactic(), syndiotactic(), TacticityMarkov(1.0, 1.0)])
def test_reset_decorrelates_isolated_units(model, rng) -> None:
    # every tactic unit isolated: chain-end control has nothing to act on
    gaps = [0] + [1] * 20_000
    configs = model.sample(gaps, rng)
    assert meso_fraction(configs, gaps, include_gapped=True) == pytest.approx(0.5, abs=0.02)


def test_carry_treats_gapped_units_as_adjacent(rng) -> None:
    gaps = [0] + [2] * 50
    assert len(set(isotactic(across_gaps="carry").sample(gaps, rng))) == 1

    alternating = syndiotactic(across_gaps="carry").sample(gaps, rng)
    assert all(a != b for a, b in zip(alternating, alternating[1:]))


def test_enantiomorphic_site_ignores_gaps() -> None:
    model = TacticityEnantiomorphicSite(0.9)
    assert model.sample([0, 5, 0, 2], np.random.default_rng(3)) == model.sample(4, np.random.default_rng(3))


def test_across_gaps_is_validated() -> None:
    with pytest.raises(ValueError, match="across_gaps"):
        TacticityBernoullian(0.5, across_gaps="sometimes")
    with pytest.raises(ValueError, match="across_gaps"):
        TacticityMarkov(0.5, 0.5, across_gaps="sometimes")


def test_negative_gaps_are_rejected(rng) -> None:
    with pytest.raises(ValueError):
        atactic().sample([0, -1], rng)


# Diad statistics with gaps
def test_diads_count_only_adjacent_pairs_by_default() -> None:
    configs = [1, 1, -1, -1]
    gaps = [0, 0, 3, 0]  # the second and third units are separated by a gap

    assert configs_to_diads(configs) == ["m", "r", "m"]  # no gaps: all adjacent
    assert configs_to_diads(configs, gaps) == ["m", "m"]  # the gapped pair is skipped
    assert configs_to_diads(configs, gaps, include_gapped=True) == ["m", "r", "m"]
    assert meso_fraction(configs, gaps) == 1.0


def test_diads_need_one_gap_per_config() -> None:
    with pytest.raises(ValueError):
        configs_to_diads([1, 1, 1], [0, 0])


def test_meso_fraction_without_diads_is_nan() -> None:
    assert np.isnan(meso_fraction([1, -1], [0, 2]))


# TacticityConstant: one config for a whole chain, across gaps too
def test_constant_holds_one_config_across_gaps(rng) -> None:
    gaps = [0, 3, 0, 1, 5, 0]
    chains = [TacticityConstant().sample(gaps, rng) for _ in range(200)]

    assert all(len(set(configs)) == 1 for configs in chains)  # one config per chain
    assert {configs[0] for configs in chains} == {1, -1}  # chosen at random between chains


@pytest.mark.parametrize("config", [1, -1])
def test_constant_with_a_fixed_config(config, rng) -> None:
    assert TacticityConstant(config).sample([0, 2, 0, 7], rng) == [config] * 4


def test_constant_rejects_other_configs() -> None:
    with pytest.raises(ValueError, match="Config"):
        TacticityConstant(0)


def test_isotactic_runs_may_invert_across_gaps(rng) -> None:
    # the documented contrast with TacticityConstant: under isotactic() (reset), each run is
    # isotactic, but the runs on either side of a gap differ in about half of chains
    gaps = [0, 0, 0, 2, 0, 0]
    flipped = [configs[0] != configs[3] for configs in (isotactic().sample(gaps, rng) for _ in range(2_000))]
    assert sum(flipped) / len(flipped) == pytest.approx(0.5, abs=0.05)


# Site control: handedness and syndioselective sites
def m_runs(configs) -> list[int]:
    """Lengths of the runs of consecutive meso diads in a chain"""
    runs, length = [], 0
    for diad in configs_to_diads(configs) + ["r"]:
        if diad == "m":
            length += 1
        elif length:
            runs.append(length)
            length = 0
    return runs


def test_racemic_sites_split_chains_between_handednesses(rng) -> None:
    # a perfectly selective site: every config in a chain is the site's handedness
    firsts = [TacticityEnantiomorphicSite(1.0).sample(5, rng)[0] for _ in range(4_000)]
    assert firsts.count(1) / len(firsts) == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("handedness", [1, -1])
def test_enantiopure_sites_fix_the_handedness(handedness, rng) -> None:
    assert TacticityEnantiomorphicSite(1.0, site_handedness=handedness).sample(20, rng) == [handedness] * 20


@pytest.mark.parametrize(
    "kwargs",
    [dict(sigma=0.3), dict(sigma=1.2), dict(sigma=0.9, p_flip=1.5), dict(sigma=0.9, site_handedness=0)],
)
def test_site_model_validates_its_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        TacticityEnantiomorphicSite(**kwargs)


def test_fixed_handedness_matches_the_original_model() -> None:
    # site_handedness=1 with no flips draws exactly like the original one-parameter model
    for seed in range(10):
        old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        old = [1 if draw < 0.8 else -1 for draw in old_rng.random(25)]
        assert TacticityEnantiomorphicSite(0.8, site_handedness=1).sample(25, new_rng) == old
        assert old_rng.bit_generator.state == new_rng.bit_generator.state


def test_perfect_syndioselective_site_alternates(rng) -> None:
    configs = syndiospecific_site().sample(50, rng)
    assert configs_to_diads(configs) == ["r"] * 49


def test_syndioselective_meso_fraction(rng) -> None:
    sigma = 0.8
    configs = syndiospecific_site(sigma).sample(20_000, rng)
    assert meso_fraction(configs) == pytest.approx(2 * sigma * (1 - sigma), abs=0.02)


def test_facial_errors_give_meso_pairs(rng) -> None:
    # an isolated wrong face at a syndioselective site: ...rr mm rr... (rmmr pentads)
    runs = [run for _ in range(200) for run in m_runs(syndiospecific_site(0.97).sample(100, rng))]
    assert runs.count(2) / len(runs) > 0.9


def test_skips_give_isolated_meso_diads(rng) -> None:
    # a back-skip at a perfectly selective site: ...rr m rr... (rmrr pentads)
    runs = [run for _ in range(200) for run in m_runs(syndiospecific_site(1.0, p_skip=0.03).sample(100, rng))]
    assert runs.count(1) / len(runs) > 0.9


@pytest.mark.parametrize("gap", [0, 1, 3])
def test_site_switches_once_per_insertion_across_gaps(gap, rng) -> None:
    # with sigma = 1 each config is the site state, so a changed config means a net switch
    model = TacticityEnantiomorphicSite(1.0, p_flip=0.3)
    changed = [a != b for a, b in (model.sample([0, gap], rng) for _ in range(20_000))]
    assert sum(changed) / len(changed) == pytest.approx(model.net_flip_probability(gap + 1), abs=0.015)


@pytest.mark.parametrize(
    "model, draws_per_call, draws_per_unit",
    [
        (TacticityEnantiomorphicSite(0.9), 1, 1),  # racemic: one handedness draw per chain
        (TacticityEnantiomorphicSite(0.9, site_handedness=-1), 0, 1),
        (TacticityEnantiomorphicSite(0.9, p_flip=0.5), 1, 2),  # switching sites draw once more per unit
        (syndiospecific_site(0.9, p_skip=0.1, site_handedness=1), 0, 2),
    ],
)
def test_site_model_draw_counts_ignore_gaps(model, draws_per_call, draws_per_unit) -> None:
    for gaps in ([0, 0, 0, 0], [0, 4, 1, 7]):
        used, reference = np.random.default_rng(3), np.random.default_rng(3)
        model.sample(gaps, used)
        reference.random(draws_per_call + draws_per_unit * len(gaps))
        assert used.bit_generator.state == reference.bit_generator.state


def test_syndiospecific_site_validates_p_skip() -> None:
    with pytest.raises(ValueError, match="p_skip"):
        syndiospecific_site(p_skip=2.0)
