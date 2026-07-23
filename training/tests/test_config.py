from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_jets_24k_config_invariants():
    config = yaml.safe_load(
        (ROOT / "espnet_recipe/conf/tuning/train_jets_uk_24k.yaml").read_text()
    )
    generator = config["tts_conf"]["generator_params"]
    mel = config["tts_conf"]["mel_loss_params"]
    assert config["tts"] == "jets"
    assert mel["fs"] == 24000
    assert mel["n_fft"] == 1024
    assert mel["hop_length"] == 256
    assert mel["n_mels"] == 80
    assert generator["generator_upsample_scales"] == [8, 8, 2, 2]
    assert __import__("math").prod(generator["generator_upsample_scales"]) == 256
    assert generator["generator_out_channels"] == 1
    assert config["pitch_extract"] == "dio"
    assert config["energy_extract"] == "energy"
    assert config["pitch_normalize"] == "global_mvn"
    assert config["energy_normalize"] == "global_mvn"
    assert config["optim"] == config["optim2"] == "adamw"
    assert config["generator_first"] is True
    assert config["use_amp"] is False
    assert config["batch_bins"] == 1_000_000


def test_multispeaker_jets_uses_ecapa_embeddings():
    config = yaml.safe_load(
        (
            ROOT
            / "espnet_recipe/conf/tuning/train_jets_uk_24k_multispeaker.yaml"
        ).read_text()
    )
    generator = config["tts_conf"]["generator_params"]
    assert config["tts"] == "jets"
    assert generator["spks"] == -1
    assert generator["spk_embed_dim"] == 192
    assert generator["spk_embed_integration_type"] == "add"
    assert config["tts_conf"]["sampling_rate"] == 24000
