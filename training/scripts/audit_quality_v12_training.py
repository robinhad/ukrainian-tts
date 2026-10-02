"""Bind the completed fixed-cascade corpus and prepared features to the 50K run."""
import argparse
import json
from pathlib import Path

from training.quality.common import file_hash, read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inputs', 'prepared', 'seal'], required=True)
    args = parser.parse_args()
    data = Path('training/data/quality_v12')
    reports = Path('training/quality_runs/v12')
    run = json.loads((data / 'run.json').read_text())
    complete = json.loads((data / 'complete.json').read_text())
    original = read_rows('training/data/quality_v10_raw/all.parquet')
    records = read_rows(data / 'records.jsonl')
    assert complete['status'] == 'complete' and complete['evaluated'] == len(original)
    assert complete['run_key'] == run['run_key'] and run['limit'] == 0 and run['threshold'] == 3.5
    assert run['profile'] == file_hash('training/conf/quality_v12_processing.yaml')
    assert run['manifest'] == file_hash('training/data/quality_v10_raw/all.parquet')
    assert complete['retained_with_heldout'] == len(records)
    original_by_id = {r['utterance_id']: r for r in original}
    heldout = {r['utterance_id'] for r in original if r['split'].endswith('_eval')}
    assert heldout == {r['utterance_id'] for r in records if r['split'].endswith('_eval')}
    assert len({r['utterance_id'] for r in records}) == len(records)
    for row in records:
        assert row['split'] == original_by_id[row['utterance_id']]['split'].replace('quality_v10_', 'quality_v12_')
        assert row['boundary_method'] == 'mfa' and 'voa' not in row['source_id'].lower()
        assert row['sigmos_overall'] >= 3.5 or row['split'].endswith('_eval')
        assert file_hash(row['audio_path']) == row['audio_sha256'], 'Training waveform changed'
    assert {r['utterance_id'] for r in read_rows('training/quality_runs/v10/panels/heldout.jsonl')} <= heldout
    inputs = {'run_key': run['run_key'], 'records_sha256': file_hash(data / 'records.jsonl'),
              'target_steps': 50000, 'embedding_policy': 'processed_audio_only',
              'config_sha256': file_hash('training/espnet_recipe/conf/tuning/train_jets_uk_24k_multispeaker.yaml')}
    binding = reports / 'training_inputs.json'
    if binding.exists():
        assert json.loads(binding.read_text()) == inputs, 'Training inputs changed'
    else:
        write_json(binding, inputs)
    if args.stage != 'inputs':
        artifacts = [data / 'manifests/all.parquet', reports / 'dataset_validation.json',
                     reports / 'embeddings.json',
                     Path('training/dump_quality_v12/token_list/phn_espeak_ng_ukrainian/tokens.txt')]
        for split in ['train', 'dev', 'eval']:
            artifacts.extend(Path('training/dump_quality_v12') / name / ('quality_v12_' + split) / filename
                             for name, filename in [('raw', 'wav.scp'), ('xvector', 'xvector.scp')])
            archives = list((Path('training/dump_quality_v12/xvector') / ('quality_v12_' + split)).rglob('*.ark'))
            assert archives, 'Missing speaker embedding archive'
            artifacts.extend(archives)
        stats = Path('training/exp_quality_v12/tts_stats_raw_phn_espeak_ng_ukrainian')
        for phase in ['train', 'valid']:
            for name in ['feats', 'pitch', 'energy']:
                artifacts.append(stats / phase / (name + '_stats.npz'))
                index = stats / phase / 'collect_feats' / (name + '.scp')
                artifacts.append(index)
                artifacts.extend(Path(line.split(maxsplit=1)[1]) for line in index.read_text().splitlines() if line.strip())
            artifacts.extend(stats / phase / name for name in ['speech_shape', 'text_shape.phn', 'spembs_shape'])
        state = {'inputs': inputs, 'files': {str(p): file_hash(p) for p in sorted(set(artifacts))}}
        assert json.loads((reports / 'dataset_validation.json').read_text())['status'] == 'PASS'
        assert json.loads((reports / 'embeddings.json').read_text())['status'] == 'PASS'
        prepared = reports / 'prepared.json'
        if args.stage == 'seal' and not prepared.exists():
            write_json(prepared, state)
        else:
            assert json.loads(prepared.read_text()) == state, 'Prepared training artifacts changed'
    print(json.dumps({'status': 'PASS', 'stage': args.stage, 'records': len(records),
                      'heldout': len(heldout), 'target_steps': 50000}))


if __name__ == '__main__':
    main()
