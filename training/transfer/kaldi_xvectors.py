"""Extract donor-compatible SITW Kaldi x-vectors for prepared Ukrainian audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess

import kaldiio
import numpy as np
import soundfile as sf

from .convert import sha256
from .prepare import read_audio


def environment(prefix):
    env = dict(os.environ)
    env['PATH'] = str(prefix.resolve() / 'bin') + ':' + env['PATH']
    env['LD_LIBRARY_PATH'] = str(prefix.resolve() / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def command(args, env, log):
    with log.open('a') as stream:
        subprocess.run(list(map(str, args)), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def extract_chunk(index, rows, output, env, network, min_chunk, max_chunk):
    root = output / f'part-{index:04d}'
    root.mkdir(exist_ok=True)
    scp = root / 'xvector.scp'
    if (root / 'complete.json').exists():
        return scp
    scratch = root / 'scratch'
    scratch.mkdir(exist_ok=True)
    entries = []
    for uid, path in rows:
        destination = scratch / f'{uid}.wav'
        sf.write(destination, read_audio(path, 16000), 16000, subtype='PCM_16')
        entries.append(f'{uid} {destination.resolve()}\n')
    (scratch / 'wav.scp').write_text(''.join(entries))
    log = root / 'extract.log'
    feats, vad, cmvn, voiced = [scratch / name for name in ['feats.ark', 'vad.ark', 'cmvn.ark', 'voiced.ark']]
    command(['compute-mfcc-feats', '--sample-frequency=16000', '--frame-length=25',
             '--low-freq=20', '--high-freq=7600', '--num-mel-bins=30', '--num-ceps=30',
             '--snip-edges=false', f'scp:{scratch / "wav.scp"}', f'ark:{feats}'], env, log)
    command(['compute-vad', '--vad-energy-threshold=5.5', '--vad-energy-mean-scale=0.5',
             '--vad-proportion-threshold=0.12', '--vad-frames-context=2',
             f'ark:{feats}', f'ark,scp:{vad},{scratch / "vad.scp"}'], env, log)
    command(['apply-cmvn-sliding', '--norm-vars=false', '--center=true', '--cmn-window=300',
             f'ark:{feats}', f'ark:{cmvn}'], env, log)
    command(['select-voiced-frames', f'ark:{cmvn}', f'scp:{scratch / "vad.scp"}', f'ark:{voiced}'], env, log)
    command(['nnet3-xvector-compute', '--use-gpu=no', f'--min-chunk-size={min_chunk}',
             f'--chunk-size={max_chunk}', network, f'ark:{voiced}',
             f'ark,scp:{root / "xvector.ark"},{scp}'], env, log)
    vectors = dict(kaldiio.load_scp(str(scp)))
    for value in vectors.values():
        if value.shape != (512,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
            raise ValueError('Invalid donor speaker vector')
    missing = sorted(set(uid for uid, _ in rows) - vectors.keys())
    (root / 'complete.json').write_text(json.dumps(dict(items=len(vectors), missing=missing)) + '\n')
    shutil.rmtree(scratch)
    return scp


def extract(prepared, model, prefix, output, workers=8, limit=0):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits, all_rows = {}, {}
    for split in ['train', 'dev']:
        rows = dict(line.split(maxsplit=1) for line in (prepared / split / 'wav.scp').read_text().splitlines())
        if limit:
            rows = dict(list(sorted(rows.items()))[:limit])
        if all_rows.keys() & rows.keys():
            raise ValueError('Train/dev IDs overlap')
        splits[split] = rows
        all_rows.update(rows)
    identity = dict(model_sha256=sha256(model / 'final.raw'), extract_config_sha256=sha256(model / 'extract.config'),
                    wav_scp_sha256={s: sha256(prepared / s / 'wav.scp') for s in splits}, limit=limit)
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Extraction inputs changed; use a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    env = environment(prefix)
    network = output / 'extractor.raw'
    command(['nnet3-copy', f'--nnet-config={model / "extract.config"}', model / 'final.raw', network],
            env, output / 'network.log')
    minimum = int((model / 'min_chunk_size').read_text())
    maximum = int((model / 'max_chunk_size').read_text())
    rows = sorted(all_rows.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(extract_chunk, i // 256, rows[i:i+256], output, env, network, minimum, maximum)
                for i in range(0, len(rows), 256)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            print(f'X-vector chunks: {done}/{len(jobs)}', flush=True)
    vectors = {}
    for part in sorted(output.glob('part-*/xvector.scp')):
        vectors.update(line.split(maxsplit=1) for line in part.read_text().splitlines())
    missing = sorted(all_rows.keys() - vectors.keys())
    if len(missing) > max(1, len(all_rows) * .01):
        raise ValueError('More than 1% of speaker vectors missing; inspect extraction logs')
    report = dict(identity=identity, dimension=512, tool='Kaldi SITW xvector system 1a',
                  device='CPU (installed Kaldi build)', mfcc_sample_rate=16000,
                  cmvn='per-recording centered 300-frame sliding mean; variance unchanged',
                  corpus_statistics_required=False, missing_ids=missing, splits={})
    for split, selected in splits.items():
        directory = output / split
        directory.mkdir(exist_ok=True)
        ids = sorted(selected.keys() & vectors.keys())
        (directory / 'xvector.scp').write_text(''.join(f'{uid} {vectors[uid]}\n' for uid in ids))
        report['splits'][split] = dict(requested=len(selected), extracted=len(ids))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'model', 'prefix', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    extract(args.prepared, args.model.resolve(), args.prefix, args.output, args.workers, args.limit)


if __name__ == '__main__':
    main()
