from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description='Report-only speech quality experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select')
    select.add_argument('--manifest', type=Path, required=True)
    select.add_argument('--output', type=Path, required=True)
    select.add_argument('--per-source', type=int, default=100)
    select.add_argument('--seed', type=int, default=777)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--panel', type=Path, required=True)
    evaluate.add_argument('--wav-dir', type=Path)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--label', required=True)
    evaluate.add_argument('--config', type=Path, default=Path('training/conf/quality.yaml'))
    evaluate.add_argument('--checkpoint', type=Path)
    evaluate.add_argument('--training-config', type=Path)
    evaluate.add_argument('--resume', action='store_true')
    evaluate.add_argument('--allow-extra-wavs', action='store_true')
    compare = sub.add_parser('compare')
    compare.add_argument('--candidate', type=Path, required=True)
    compare.add_argument('--original', type=Path, required=True)
    compare.add_argument('--previous', type=Path)
    compare.add_argument('--best', type=Path)
    compare.add_argument('--output', type=Path, required=True)
    process = sub.add_parser('process')
    process.add_argument('--panel', type=Path, required=True)
    process.add_argument('--profile', type=Path, required=True)
    process.add_argument('--output', type=Path, required=True)
    process.add_argument('--device', default='cuda')
    process.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'select':
        from .selection import select
        select(args.manifest, args.output, args.per_source, args.seed)
    elif args.command == 'evaluate':
        from .backends import Models
        from .evaluate import evaluate
        config = yaml.safe_load(args.config.read_text())
        evaluate(args.panel, args.output, args.label, config, Models(config['models']),
                 args.wav_dir, args.checkpoint, args.training_config, args.resume, args.allow_extra_wavs)
    elif args.command == 'compare':
        from .compare import compare
        compare(args.candidate, {k: v for k in ('original', 'previous', 'best')
                                if (v := getattr(args, k)) is not None}, args.output)
    elif args.command == 'process':
        from .processing import process
        process(args.panel, args.profile, args.output, args.device, args.resume)


if __name__ == '__main__':
    main()
