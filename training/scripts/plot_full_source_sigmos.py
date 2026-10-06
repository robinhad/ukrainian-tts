"""Full-corpus processed SigMOS distributions and training-selection hours by source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from training.scripts.listening_dataset_stats import SOURCE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, default=Path('training/reports/quality_v12_scores.jsonl'))
    parser.add_argument('--train-manifest', type=Path, default=Path('training/data/quality_v12/manifests/quality_v12_train.parquet'))
    parser.add_argument('--output-prefix', type=Path, default=Path('training/reports/quality_v12_full_source_sigmos'))
    args = parser.parse_args()
    data = pd.read_json(args.scores, lines=True)
    required = ['sample_id', 'source_id', 'sigmos_overall', 'duration_seconds']
    if data[required].isna().any().any() or data.sample_id.duplicated().any():
        raise ValueError('Missing fields or duplicate recordings')
    if not np.isfinite(data.sigmos_overall).all() or not data.sigmos_overall.between(1, 5).all():
        raise ValueError('Invalid SigMOS')
    if not np.isfinite(data.duration_seconds).all() or (data.duration_seconds <= 0).any():
        raise ValueError('Invalid duration')
    train = pd.read_parquet(args.train_manifest)
    if not train['split'].str.endswith('_train').all() or train.utterance_id.duplicated().any():
        raise ValueError('Training manifest must contain unique training items')
    ids = {hashlib.sha256(json.dumps((r.source_id, r.utterance_id)).encode()).hexdigest()[:16]: r
           for r in train.itertuples()}
    if len(ids) != len(train) or not set(ids).issubset(set(data.sample_id)):
        raise ValueError('Training IDs do not match full-corpus score IDs')
    selected = data[data.sample_id.isin(ids)]
    for r in selected.itertuples():
        record = ids[r.sample_id]
        if r.heldout or r.sigmos_overall < 3.5 or not r.passed_threshold or r.source_id != record.source_id or abs(r.duration_seconds-record.duration) > 1e-7:
            raise ValueError('Selection does not match the scored training audio')
    sources = sorted(data.source_id.unique(), key=lambda k: (-data.loc[data.source_id.eq(k), 'sigmos_overall'].median(), k))
    edges = np.linspace(1, 5, 41)
    groups = [('full', data), ('train', selected)]
    summaries, bins = [], []
    for scope, population in groups:
        for source in ['all', *sources]:
            rows = population if source == 'all' else population[population.source_id.eq(source)]
            values = rows.sigmos_overall.to_numpy()
            hist, _ = np.histogram(values, edges)
            assert hist.sum() == len(rows)
            summaries.append({'population': scope, 'source_id': source, 'source': SOURCE_NAMES.get(source, 'All sources'),
                              'recordings': len(rows), 'hours': float(rows.duration_seconds.sum()/3600),
                              'median_sigmos': float(np.median(values)), 'mean_sigmos': float(values.mean()),
                              'stddev_population': float(values.std()), 'p05': float(np.quantile(values,.05)),
                              'p25': float(np.quantile(values,.25)), 'p75': float(np.quantile(values,.75)),
                              'p95': float(np.quantile(values,.95)), 'count_ge3_5': int((values>=3.5).sum()),
                              'percent_ge3_5': float((values>=3.5).mean()*100)})
            for lo, hi, count in zip(edges[:-1], edges[1:], hist):
                bins.append({'population': scope, 'source_id': source, 'lower': float(lo), 'upper': float(hi),
                             'count': int(count), 'percent': float(count/len(rows)*100)})
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'metric': 'SigMOS overall', 'audio': 'V12: MFA + ClearVoice → Sidon → DeepFilterNet3, 100% wet',
                'full_scope': 'All non-VOA processed recordings before filtering, including held-out evaluation',
                'train_scope': 'Exact training manifest after >=3.5 selection; development/evaluation excluded',
                'weighting': 'One recording, one observation; histogram percentages normalized within each source and population',
                'bin_width': 0.1, 'bin_policy': 'Left inclusive, right exclusive; final bin includes 5',
                'scores_sha256': hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                'train_manifest_sha256': hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
                'summaries': summaries}
    Path(str(prefix)+'_summary.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    for suffix, rows in [('summary', summaries), ('bins', bins)]:
        with Path(str(prefix)+f'_{suffix}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'], 'font.size':11,
                         'figure.facecolor':'#fffff8', 'axes.facecolor':'#fffff8', 'text.color':'#111',
                         'axes.labelcolor':'#333', 'axes.edgecolor':'#999', 'axes.linewidth':.6,
                         'axes.spines.top':False, 'axes.spines.right':False, 'axes.grid':False,
                         'xtick.color':'#555', 'ytick.color':'#555', 'xtick.labelsize':10,'ytick.labelsize':10})
    for scope, population in groups:
        ymax = np.ceil(max(r['percent'] for r in bins if r['source_id'] != 'all' and r['population'] == scope)/5)*5
        total = next(r for r in summaries if r['population']==scope and r['source_id']=='all')
        fig, axes = plt.subplots(4, 2, figsize=(12, 11.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=.075, right=.975, top=.83, bottom=.10, hspace=.85, wspace=.18)
        if scope == 'full':
            title = f"Only {total['percent_ge3_5']:.1f}% of processed recordings reach SigMOS 3.5"
            subtitle = f"Full non-VOA corpus · {total['recordings']:,} recordings · {total['hours']:.2f} processed hours · before filtering"
            color = '#666'
        else:
            title = f"The ≥3.5 selection leaves {total['hours']:.2f} training hours"
            subtitle = f"Exact training split · {total['recordings']:,} recordings · development and evaluation excluded"
            color = '#666'
        fig.text(.075,.965,title,fontsize=20,va='top')
        fig.text(.075,.927,subtitle,fontsize=12,va='top')
        fig.text(.075,.898,'MFA + ClearVoice → Sidon → DeepFilterNet3 · each recording has equal weight',fontsize=11,va='top',color='#555')
        for ax, source in zip(axes.flat, sources):
            summary = next(r for r in summaries if r['population']==scope and r['source_id']==source)
            counts, _ = np.histogram(population.loc[population.source_id.eq(source),'sigmos_overall'],edges)
            percent = counts/summary['recordings']*100
            ax.stairs(percent,edges,fill=True,color=color,linewidth=.7,alpha=.85)
            ax.axvline(3.5,color='#a63e25',lw=1,ls=(0,(3,3)))
            if scope == 'full':
                ax.text(3.55,ymax*.86,'3.5 cutoff',fontsize=9,color='#a63e25')
            ax.set_title(SOURCE_NAMES.get(source,source),loc='left',fontsize=14,pad=31)
            ax.text(0,1.07,f"{summary['recordings']:,} recordings · {summary['hours']:.2f} h · median {summary['median_sigmos']:.3f}",
                    transform=ax.transAxes,fontsize=10.5,va='bottom')
            xmin = 1 if scope == 'full' else 3.5
            ax.set_xlim(xmin,5);ax.set_ylim(0,ymax);ax.set_xticks([1,2,3,4,5] if scope == 'full' else [3.5,4,4.5,5])
            ax.set_yticks(np.arange(0,ymax+1,5 if ymax<=25 else 20));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
            ax.spines['bottom'].set_bounds(xmin,5);ax.spines['left'].set_bounds(0,ymax)
            ax.tick_params(length=3,width=.6)
            for tick in [*ax.get_xticklabels(),*ax.get_yticklabels()]:tick.set_fontfamily('DejaVu Sans')
        fig.supylabel('Recordings within each source (%)',x=.018,fontsize=12)
        fig.supxlabel('Overall SigMOS (higher is better)',y=.055,fontsize=12)
        fig.text(.075,.018,'0.1-point bins; shared axes within each figure. The training figure zooms to ≥3.5 and uses its own percentage scale.',fontsize=9.5,color='#555')
        for ext in ['png','pdf']:
            fig.savefig(str(prefix)+f'_{scope}.{ext}',dpi=170,facecolor='#fffff8',bbox_inches='tight')
        plt.close(fig)
    print(json.dumps([r for r in summaries if r['source_id']=='all']))


if __name__ == '__main__':
    main()
