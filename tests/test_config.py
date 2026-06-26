"""Config (de)serialisation and ablation-flag behaviour."""

import dataclasses

from acdan.config import ACDANConfig, AblationFlags, baseline_cot_config


def test_from_dict_partial_uses_defaults():
    cfg = ACDANConfig.from_dict({"name": "x", "seed": 3, "dto": {"iters": 5}})
    assert cfg.name == "x"
    assert cfg.seed == 3
    assert cfg.dto.iters == 5
    # untouched sub-config keeps its default
    assert cfg.latent.recurrent_depth == ACDANConfig().latent.recurrent_depth


def test_to_dict_roundtrip():
    cfg = ACDANConfig(name="rt", seed=9)
    again = ACDANConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_yaml_roundtrip(tmp_path):
    cfg = ACDANConfig(name="yaml", seed=2)
    p = tmp_path / "c.yaml"
    cfg.save_yaml(str(p))
    loaded = ACDANConfig.from_yaml(str(p))
    assert loaded.to_dict() == cfg.to_dict()


def test_baseline_disables_everything():
    flags = baseline_cot_config().ablation
    assert not any(dataclasses.asdict(flags).values())


def test_describe_lists_modules():
    desc = AblationFlags().describe()
    assert "enabled=" in desc and "disabled=" in desc
