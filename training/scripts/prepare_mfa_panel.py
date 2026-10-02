"""Prepare untrimmed mono audio and transcript vocabulary for the fixed MFA pilot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import soundfile as sf

from training.frontend.sanitize import sanitize_text
from training.quality.common import file_hash, read_rows, write_json, write_tables
from training.scripts.materialize_non_voa import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('training/quality_runs/v10/panels/processing.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/mfa_trim'))
    args = parser.parse_args()
    rows = read_rows(args.panel)
    if len(rows) != 800 or sorted(Counter(r['source_id'] for r in rows).values()) != [100] * 8:
        raise ValueError('Expected the fixed source-balanced 800-recording panel')
    def prepare(row):
        source = Path(row['audio_path'])
        if file_hash(source) != row['reference_sha256']:
            raise ValueError('Frozen native reference changed')
        identifier = hashlib.sha256(json.dumps((row['source_id'], row['utterance_id'])).encode()).hexdigest()[:16]
        speaker = hashlib.sha256((row['source_id'] + '/' + row['speaker_id']).encode()).hexdigest()[:16]
        target = args.output / 'corpus' / speaker / (identifier + '.wav')
        target.parent.mkdir(parents=True, exist_ok=True)
        audio = decode(source.read_bytes())
        sf.write(target, audio, 24000, subtype='PCM_24')
        text = sanitize_text(row['text']).lower().replace('\u0301', '')
        words = re.findall(r"[\w']+", text, flags=re.UNICODE)
        words = [w.strip("'") for w in words if w.strip("'")]
        unsupported = [w for w in words if not re.fullmatch("[а-щьюяєіїґ']+", w)]
        target.with_suffix('.lab').write_text(' '.join(words) + '\n')
        return {**row, 'sample_id': identifier, 'alignment_audio_path': str(target.resolve()),
                'alignment_audio_sha256': file_hash(target), 'alignment_duration': len(audio) / 24000,
                'alignment_text': ' '.join(words), 'unsupported_tokens': unsupported}
    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(prepare, rows))
    write_tables(args.output, 'panel', prepared)
    dictionary = args.output / 'models/ukrainian_mfa_dictionary.dict'
    known = {line.split()[0] for line in dictionary.read_text().splitlines() if line.strip()}
    vocabulary = {word for r in prepared for word in r['alignment_text'].split() if re.fullmatch("[а-щьюяєіїґ']+", word)}
    oovs = sorted(vocabulary - known)
    (args.output / 'oovs.txt').write_text('\n'.join(oovs) + '\n')
    write_json(args.output / 'prepare_summary.json', {'count': len(prepared), 'vocabulary': len(vocabulary),
               'oov_words': len(oovs), 'unsupported_transcript_items': sum(bool(r['unsupported_tokens']) for r in prepared)})
    print(f'Prepared {len(prepared)} untrimmed recordings; {len(oovs)} OOV words', flush=True)


if __name__ == '__main__':
    main()
