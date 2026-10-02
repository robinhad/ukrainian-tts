"""Build exact native/selected listening comparisons for brute-force SigMOS winners."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_selected_audio_listening import page, SOURCES, read_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path('training/reports/quality_bruteforce_selection.jsonl'))
    parser.add_argument('--search', type=Path, default=Path('training/quality_runs/v10/search'))
    parser.add_argument('--normalized', type=Path, default=Path('training/quality_runs/normalized_original/per_file.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/brute_force/listening'))
    args = parser.parse_args()
    decisions = [json.loads(line) for line in args.selection.read_text().splitlines()]
    if len(decisions) != 800 or len({r['sample_id'] for r in decisions}) != 800:
        raise ValueError('Expected 800 unique decisions')
    panels = {}
    for variant in {'original'} | {r['selected_variant'] for r in decisions}:
        path = args.normalized if variant == 'normalized_original' else args.search / variant / ('per_file.jsonl' if variant == 'original' else 'quality/per_file.jsonl')
        panels[variant] = {hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]: row for key, row in read_panel(path).items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audio').mkdir(exist_ok=True)
    rows = []
    def copy(row, relative):
        source = Path(row['audio_path'])
        expected = row['audio_sha256']
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Scored audio changed')
        shutil.copyfile(source, args.output / relative)
        if hashlib.sha256((args.output / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Copied audio changed')
    for decision in decisions:
        identifier = decision['sample_id']
        variant = decision['selected_variant']
        selected, original = panels[variant][identifier], panels['original'][identifier]
        if (selected['sigmos_overall'] != decision['selected_sigmos_overall']
                or selected['audio_sha256'] != decision['selected_audio_sha256']
                or selected['source_id'] != decision['source_id']
                or selected['reference_sha256'] != original['reference_sha256']
                or decision['retained'] != (selected['sigmos_overall'] >= 3.5)):
            raise ValueError('Listening decision does not match scored file')
        if not decision['retained']:
            continue
        url, reference_url = f'audio/{identifier}.wav', f'audio/{identifier}_original.wav'
        copy(selected, url)
        copy(original, reference_url)
        rows.append({'sample_id': identifier, 'source_id': decision['source_id'],
                     'selected_variant': variant, 'selected_score': selected['sigmos_overall'],
                     'original_score': original['sigmos_overall'], 'seconds': selected['duration_seconds'],
                     'text': original['text'], 'url': url, 'reference_url': reference_url,
                     'audio_sha256': selected['audio_sha256'], 'reference_audio_sha256': original['audio_sha256']})
    rows.sort(key=lambda r: (r['source_id'], r['selected_score'], r['sample_id']))
    if not rows:
        raise ValueError('No retained recordings')
    (args.output / 'index.html').write_text(page(rows, preview=False, baseline='brute_force'))
    (args.output / 'selection.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    preview = []
    for source in SOURCES:
        candidates = [r for r in rows if r['source_id'] == source]
        count = min(4, len(candidates))
        for i in range(count):
            row = dict(candidates[round(i * (len(candidates) - 1) / max(1, count - 1))])
            for key in ['url', 'reference_url']:
                row[key] = 'data:audio/wav;base64,' + base64.b64encode((args.output / row[key]).read_bytes()).decode()
            preview.append(row)
    (args.output / 'preview.html').write_text(page(preview, preview=True, baseline='brute_force'))
    with zipfile.ZipFile(args.output / 'listening_set.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        names = ['index.html', 'preview.html', 'selection.jsonl'] + [r[k] for r in rows for k in ['url', 'reference_url']]
        for name in names:
            archive.write(args.output / name, name)
    report = {'recordings': len(rows), 'preview_recordings': len(preview), 'audio_files': len(rows) * 2,
              'selected_variants': dict(Counter(r['selected_variant'] for r in rows)),
              'sources': dict(Counter(r['source_id'] for r in rows)), 'exact_scored_files_verified': True}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
