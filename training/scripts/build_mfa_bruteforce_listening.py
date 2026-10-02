"""Build exact MFA listening pairs for brute-force selection or the best single method."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import zipfile

from build_selected_audio_listening import page, read_panel, SOURCES


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(run, selection, output, mode='brute_force'):
    if mode not in {'brute_force', 'best_method'}:
        raise ValueError('Unknown listening mode')
    completed = json.loads((run / 'complete.json').read_text())
    if completed['status'] != 'complete' or len(completed['profiles']) != 16:
        raise ValueError('Expected the completed 16-variant MFA sweep')
    panels = {name: {row['sample_id']: row for row in read_panel(run / name / 'per_file.jsonl').values()}
              for name in completed['profiles']}
    original = panels['normalized_original']
    transcripts = {r['sample_id']: r for r in map(json.loads, (run / 'panel.jsonl').read_text().splitlines())}
    best_method = max((name for name in completed['profiles'] if name not in {'identity_wet1', 'normalized_original'}),
                      key=lambda name: statistics.median(r['sigmos_overall'] for r in panels[name].values()))
    if mode == 'best_method':
        decisions = [{**r, 'selected_profile': best_method, 'retained': r['sigmos_overall'] >= 3.5}
                     for r in panels[best_method].values()]
    else:
        decisions = [r for r in map(json.loads, selection.read_text().splitlines()) if r['policy'] == 'brute_force']
    if (len(decisions) != 800 or len({r['sample_id'] for r in decisions}) != 800
            or {r['sample_id'] for r in decisions} != original.keys()
            or transcripts.keys() != original.keys()
            or any(p.keys() != original.keys() for p in panels.values())):
        raise ValueError('Selections, scores, and transcripts must cover the same 800 recordings')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audio').mkdir(exist_ok=True)
    rows = []
    priority = ([best_method] if mode == 'best_method' else
                ['normalized_original'] + sorted(set(panels) - {'normalized_original'}))
    for decision in decisions:
        identifier = decision['sample_id']
        if not re.fullmatch(r'[0-9a-f]{16}', identifier):
            raise ValueError('Invalid sample ID')
        winner = max(priority, key=lambda name: panels[name][identifier]['sigmos_overall'])
        selected, baseline, transcript = panels[winner][identifier], original[identifier], transcripts[identifier]
        if (winner != decision['selected_profile']
                or selected['sigmos_overall'] != decision['sigmos_overall']
                or selected['audio_sha256'] != decision['audio_sha256']
                or decision['retained'] != (selected['sigmos_overall'] >= 3.5)
                or selected['source_id'] != decision['source_id']
                or selected['reference_sha256'] != baseline['reference_sha256']
                or selected['processing_input_sha256'] != baseline['processing_input_sha256']
                or selected['processing_input_sha256'] != transcript['processing_input_sha256']
                or selected['reference_sha256'] != transcript['reference_sha256']
                or selected['utterance_id'] != transcript['utterance_id']
                or selected['boundary_method'] != 'mfa' or baseline['boundary_method'] != 'mfa'):
            raise ValueError('Listening selection differs from the scored MFA selection')
        urls = {}
        for key, name, scored in [('url', winner, selected), ('reference_url', 'normalized_original', baseline)]:
            source = run / name / 'wav' / (scored['utterance_id'] + '.wav')
            relative = f'audio/{identifier}_{key}.wav'
            if checksum(source) != scored['audio_sha256']:
                raise ValueError('Scored audio changed')
            shutil.copyfile(source, output / relative)
            if checksum(output / relative) != scored['audio_sha256']:
                raise ValueError('Copied audio changed')
            urls[key] = relative
        rows.append({'sample_id': identifier, 'source_id': selected['source_id'],
                     'selected_variant': winner, 'selected_score': selected['sigmos_overall'],
                     'original_score': baseline['sigmos_overall'], 'seconds': selected['duration_seconds'],
                     'text': transcript['text'], 'review_flags': transcript['mfa_flags'],
                     'retained': decision['retained'], 'audio_sha256': selected['audio_sha256'],
                     'reference_audio_sha256': baseline['audio_sha256'], **urls})
    rows.sort(key=lambda row: (row['source_id'], row['selected_score'], row['sample_id']))
    baseline_mode = 'mfa_best_method' if mode == 'best_method' else 'mfa_brute_force'
    (output / 'index.html').write_text(page(rows, preview=False, baseline=baseline_mode))
    (output / 'selection.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    preview = []
    for source in SOURCES:
        candidates = [r for r in rows if r['source_id'] == source and (mode == 'best_method' or r['retained'])]
        count = min(4, len(candidates))
        for i in range(count):
            row = dict(candidates[round(i * (len(candidates) - 1) / max(1, count - 1))])
            for key in ['url', 'reference_url']:
                row[key] = 'data:audio/wav;base64,' + base64.b64encode((output / row[key]).read_bytes()).decode()
            preview.append(row)
    (output / 'preview.html').write_text(page(preview, preview=True, baseline=baseline_mode))
    report = {'recordings': len(rows), 'retained_ge3_5': sum(r['retained'] for r in rows),
              'preview_recordings': len(preview), 'audio_files': len(rows) * 2,
              'comparison': f'normalized_original_MFA_vs_{mode}_MFA',
              'best_single_method': best_method,
              'median_processed_sigmos': statistics.median(r['selected_score'] for r in rows),
              'improved_vs_normalized': sum(r['selected_score'] > r['original_score'] for r in rows),
              'worse_vs_normalized': sum(r['selected_score'] < r['original_score'] for r in rows),
              'selected_variants': dict(Counter(r['selected_variant'] for r in rows)),
              'sources': dict(Counter(r['source_id'] for r in rows)),
              'retained_sources': dict(Counter(r['source_id'] for r in rows if r['retained'])),
              'review_recordings': sum(bool(r['review_flags']) for r in rows),
              'exact_scored_files_verified': True}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with zipfile.ZipFile(output / 'listening_set.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['index.html', 'preview.html', 'selection.jsonl', 'report.json'] + [
                r[k] for r in rows for k in ['url', 'reference_url']]:
            archive.write(output / name, name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/mfa_processing'))
    parser.add_argument('--selection', type=Path, default=Path('training/reports/quality_mfa_processing_selection.jsonl'))
    parser.add_argument('--mode', choices=['brute_force', 'best_method'], default='brute_force')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run / ('listening_best_method' if args.mode == 'best_method' else 'listening')
    print(json.dumps(build(args.run, args.selection, output, args.mode)), flush=True)


if __name__ == '__main__':
    main()
