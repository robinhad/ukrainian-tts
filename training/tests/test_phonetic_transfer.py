import json
from pathlib import Path

import pytest
import torch

from training.transfer.convert import initialize_rows
from training.transfer.frontend import TransferFrontend
from training.transfer.mapping import Features, english_ipa, map_tokens


@pytest.fixture
def features(tmp_path):
    path = tmp_path / 'features.tsv'
    path.write_text('segment\tsyllabic\tnasal\tcoronal\tvoice\n'
                    'n\t-\t+\t+\t+\n'
                    'ɲ\t-\t+\t-\t+\n'
                    'a\t+\t-\t-\t+\n')
    return Features(path)


def test_arpabet_stress_and_reduced_vowels():
    assert english_ipa('AH0') == 'ə'
    assert english_ipa('AH1') == 'ˈʌ'
    assert english_ipa('ER0') == 'ɚ'
    assert english_ipa('IY2') == 'ˌi'
    assert english_ipa('CH') == 'tʃ'
    assert english_ipa(',') is None


def test_mapping_sound_stress_specials_and_missing(features):
    source = ['<blank>', '<unk>', 'IY1', 'IY0', 'N', ',', "'", '<sos/eos>']
    target = ['<blank>', '<unk>', 'ˈi', 'i', 'ɲ', 'xʲ', ',', '«', '—', '<sos/eos>']
    rows = map_tokens(source, target, features, .3)
    assert [r['source_token'] for r in rows[:5]] == ['<blank>', '<unk>', 'IY1', 'IY0', 'N']
    assert rows[4]['method'] == 'phoible'
    assert rows[4]['distance'] == .25
    assert rows[5]['method'] == rows[8]['method'] == 'random'
    assert rows[7]['source_token'] == "'"
    assert map_tokens(source, ['ɲ'], features, .1)[0]['method'] == 'random'
    with pytest.raises(ValueError, match='Duplicate'):
        map_tokens(source, ['i', 'i'], features)


def test_independent_rows_deterministic_random(features):
    rows = map_tokens(['IY1'], ['i', 'ˈi', 'x', 'xʲ'], features)
    source = torch.arange(4).float().reshape(1, 4)
    converted = initialize_rows(source, rows, 7)
    assert torch.equal(converted, initialize_rows(source, rows, 7))
    assert torch.equal(converted[0], converted[1])
    assert not torch.equal(converted[2], converted[3])
    embedding = torch.nn.Embedding.from_pretrained(converted, freeze=False)
    optimizer = torch.optim.SGD(embedding.parameters(), lr=.1)
    embedding(torch.tensor([0])).sum().backward()
    optimizer.step()
    assert not torch.equal(embedding.weight[0], embedding.weight[1])
    assert torch.equal(embedding.weight[1], source[0])


def test_real_frontend_phone_boundaries_stress_and_ids():
    frontend = TransferFrontend(['<blank>', '<unk>', '<sos/eos>'])
    phones = frontend.phonemes('Привіт, Україно!')
    assert 'ˈiː' in phones and 't̪' in phones
    assert ',' in phones and '!' in phones
    assert frontend.phonemes('за́мок') != frontend.phonemes('замо́к')
    assert 'ˈa' in frontend.phonemes('за́мок')
    assert 'ˈo' in frontend.phonemes('замо́к')
    frontend = TransferFrontend(['<blank>', '<unk>'] + sorted(set(phones)) + ['<sos/eos>'])
    encoded, ids = frontend.encode('Привіт, Україно!')
    assert encoded == [frontend.tokens[i] for i in ids]
    with pytest.raises(ValueError, match='missing'):
        frontend.encode('Щастя')
    with pytest.raises(ValueError, match='one lexical'):
        frontend.phonemes('за́мо́к')


def test_snapshot_rejects_inventory_change(tmp_path):
    frontend = TransferFrontend(['<blank>', '<unk>', 'a', '<sos/eos>'])
    path = tmp_path / 'frontend.json'
    path.write_text(json.dumps(frontend.snapshot()))
    frontend.verify(path)
    frontend.tokens[2] = 'ɑ'
    with pytest.raises(ValueError, match='differs'):
        frontend.verify(path)


def test_monitor_reports_progress_rate_and_kyiv_eta(tmp_path):
    from datetime import datetime
    from training.transfer.monitor import snapshot
    log = tmp_path / 'train.log'
    log.write_text('2026-10-20 10:00:00,000 INFO: 1epoch:train:1-10batch\n'
                   '2026-10-20 10:00:20,000 INFO: 1epoch:train:11-20batch: generator_loss=10.5\n')
    status = snapshot(log, 20000)
    assert status['completed_steps'] == 20
    assert status['seconds_per_step'] == 2
    assert status['metrics']['generator_loss'] == 10.5
    elapsed = datetime.fromisoformat(status['eta_kyiv']) - datetime.fromisoformat(status['timestamp_kyiv'])
    assert elapsed.total_seconds() == 39960


def test_launcher_uses_espnet_cmd_keyword_and_exact_step_budget(tmp_path, monkeypatch):
    import sys
    import yaml
    from espnet2.tasks.gan_tts import GANTTSTask
    from training.transfer import train
    output = tmp_path / 'run'
    monkeypatch.setattr(train, 'training_config', lambda *args: {
        'token_list': ['<blank>', '<unk>', '<sos/eos>'],
        'output_dir': str(output), 'resume': False,
    })
    captured = {}

    # ESPnet's first positional argument is Namespace, not CLI arguments.
    def launch(*, cmd):
        captured.update(yaml.safe_load(Path(cmd[1]).read_text()))

    monkeypatch.setattr(GANTTSTask, 'main', launch)
    monkeypatch.setattr(sys, 'argv', ['train', '--bundle', str(tmp_path),
                        '--train', str(tmp_path), '--valid', str(tmp_path),
                        '--output', str(output), '--steps', '20000',
                        '--batch-size', '8', '--external-metrics'])
    train.main()
    assert captured['max_epoch'] * captured['num_iters_per_epoch'] == 20000
    assert captured['batch_size'] == 8
    assert captured['resume'] is False and captured['use_wandb'] is False


@pytest.mark.parametrize('format', ['jsonl', 'parquet'])
def test_preparation_resampling_and_training_contract(tmp_path, format):
    import numpy as np
    import soundfile as sf
    import yaml
    from training.transfer.prepare import prepare
    from training.transfer.train import training_config

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    config = yaml.safe_load(Path('training/transfer/ukrainian_inventory.yaml').read_text())
    config['feats_extract_conf'] = {'fs': 22050}
    for key in ('normalize_conf', 'pitch_normalize_conf', 'energy_normalize_conf'):
        config[key] = {'stats_file': 'stats/placeholder.npz'}
    (bundle / 'config.yaml').write_text(yaml.safe_dump(config))
    frontend = TransferFrontend(config['token_list'])
    (bundle / 'frontend.json').write_text(json.dumps(frontend.snapshot()))
    source_audio = tmp_path / 'source.wav'
    audio = .1 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
    sf.write(source_audio, audio, 48000, subtype='FLOAT')
    pairs = tmp_path / ('pairs.' + format)
    if format == 'parquet':
        import pandas as pd
        pd.DataFrame([dict(utterance_id='sample', text_sanitized='Привіт, Україно!',
                           audio_path=str(source_audio))]).to_parquet(pairs)
    else:
        pairs.write_text(json.dumps(dict(id='sample', text='Привіт, Україно!', audio_path=str(source_audio))))
    prepared = tmp_path / 'prepared'
    prepare(bundle, pairs, prepared)
    _, path = (prepared / 'wav.scp').read_text().strip().split(' ', 1)
    result, rate = sf.read(path)
    assert rate == 22050 and len(result) == 22050
    assert (prepared / 'text').read_text().strip().split(' ', 1)[1].split() == frontend.phonemes('Привіт, Україно!')
    config = training_config(bundle, prepared, prepared, tmp_path / 'run')
    assert config['resume'] is False and config['freeze_param'] == []
    assert config['init_param'] == [str((bundle / 'model.pth').resolve())]
    assert config['train_shape_file'] == [str((prepared / 'text').resolve())]
    with pytest.raises(ValueError, match='single voice'):
        training_config(bundle, prepared, prepared, tmp_path / 'run', speaker='tetiana')
    (prepared / 'text').write_text('sample corrupted\n')
    with pytest.raises(ValueError, match='modified'):
        training_config(bundle, prepared, prepared, tmp_path / 'run')


def test_select_voice_qualifies_source_and_preserves_split(tmp_path):
    import pandas as pd
    from training.transfer.convert import sha256
    from training.transfer.select_voice import select_voice
    manifests = tmp_path / 'manifests'
    prepared = tmp_path / 'prepared'
    manifests.mkdir()
    for split in ('train', 'dev'):
        directory = prepared / split
        directory.mkdir(parents=True)
        ids = [split + '-wanted', split + '-other-source', split + '-other-speaker']
        pd.DataFrame(dict(
            utterance_id=ids, speaker_id=['tetiana', 'tetiana', 'mykyta'],
            source_id=['opentts_tetiana', 'other', 'opentts_mykyta'],
            audio_sha256=ids, duration=[10., 20., 30.],
        )).to_parquet(manifests / f'quality_v12_{split}.parquet')
        (directory / 'text').write_text(''.join(f'{uid} a b\n' for uid in ids))
        (directory / 'wav.scp').write_text(''.join(f'{uid} original/{uid}.wav\n' for uid in ids))
        (directory / 'metadata.json').write_text(json.dumps(dict(
            items=3, frontend_sha256='fixed', sample_rate=22050,
            text_sha256=sha256(directory / 'text'), wav_scp_sha256=sha256(directory / 'wav.scp'))))
    output = tmp_path / 'selected'
    select_voice(manifests, prepared, output, 'tetiana', 'opentts_tetiana')
    for split in ('train', 'dev'):
        assert (output / split / 'text').read_text() == f'{split}-wanted a b\n'
        assert (output / split / 'wav.scp').read_text() == f'{split}-wanted original/{split}-wanted.wav\n'
        metadata = json.loads((output / split / 'metadata.json').read_text())
        assert metadata['items'] == 1 and metadata['speaker_id'] == 'tetiana'
        assert metadata['source_id'] == 'opentts_tetiana'
