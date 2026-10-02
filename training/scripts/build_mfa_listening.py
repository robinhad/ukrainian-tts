"""Build 32 native/current-trim/MFA listening examples, including review cases."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_selected_audio_listening import page, SOURCES, read_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/mfa_trim'))
    args = parser.parse_args()
    output = args.run / 'listening'
    (output / 'audio').mkdir(parents=True, exist_ok=True)
    candidates = read_panel(args.run / 'quality/per_file.jsonl')
    original = read_panel(Path('training/quality_runs/v10/search/original/per_file.jsonl'))
    energy = read_panel(Path('training/quality_runs/v10/search/identity_wet1/quality/per_file.jsonl'))
    metadata = {(r['source_id'], r['utterance_id']): r for r in map(json.loads, (args.run / 'processing.jsonl').read_text().splitlines())}
    chosen = []
    for source in SOURCES:
        keys = [key for key in candidates if key[0] == source]
        ranked = sorted(keys, key=lambda k: (-metadata[k]['removed_seconds'], k))
        review = sorted(keys, key=lambda k: (-bool(metadata[k]['flags']), k))
        asr = sorted(keys, key=lambda k: (-(candidates[k]['wer'] - energy[k]['wer']), k))
        selection = []
        for ranking in [ranked, review, asr, ranked]:
            selection.append(next(k for k in ranking if k not in selection))
        chosen.extend(selection)
    rows = []
    for key in chosen:
        row, meta = candidates[key], metadata[key]
        if row['audio_sha256'] != meta['output_sha256']:
            raise ValueError('MFA score/render mismatch')
        identifier = meta['sample_id']
        urls = {}
        for name, candidate in [('url', row), ('reference_url', original[key]), ('energy_url', energy[key])]:
            source = Path(candidate['audio_path'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != candidate['audio_sha256']:
                raise ValueError('Scored audio changed')
            relative = f'audio/{identifier}_{name}.wav'
            shutil.copyfile(source, output / relative)
            urls[name] = relative
        rows.append({'sample_id': identifier, 'source_id': key[0], 'selected_variant': 'mfa_trim',
                     'selected_score': row['sigmos_overall'], 'original_score': original[key]['sigmos_overall'],
                     'energy_score': energy[key]['sigmos_overall'], 'seconds': row['duration_seconds'],
                     'text': original[key]['text'], 'review_flags': meta['flags'], **urls})
    (output / 'index.html').write_text(page(rows, preview=False, baseline='mfa'))
    embedded = []
    for row in rows:
        row = dict(row)
        for name in ['url', 'reference_url', 'energy_url']:
            row[name] = 'data:audio/wav;base64,' + base64.b64encode((output / row[name]).read_bytes()).decode()
        embedded.append(row)
    (output / 'preview.html').write_text(page(embedded, preview=True, baseline='mfa'))
    (output / 'selection.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with zipfile.ZipFile(output / 'listening_set.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['index.html', 'preview.html', 'selection.jsonl'] + [r[k] for r in rows for k in ['url', 'reference_url', 'energy_url']]:
            archive.write(output / name, name)
    print(f'Built {len(rows)} three-way listening examples', flush=True)


if __name__ == '__main__':
    main()
