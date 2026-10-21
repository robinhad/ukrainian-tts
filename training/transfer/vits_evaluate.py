"""Fixed multi-source VITS listening panel, phonetic diagnostics and MOS metrics."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path

import kaldiio
import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .convert import sha256
from .rhotic_diagnostic import score
from .rhotic_report import summary
from .vits import load
from training.scripts.score_checkpoint_listening import QualityModels


def evaluate(root, step):
    record = json.loads((root/'experiment.json').read_text())
    # Checkpoints are saved every 1K steps; on-demand listening need not wait
    # for the automatic 5K evaluation interval.
    if step < 0 or step > record['steps'] or step % 1000:
        raise ValueError('Evaluation step must match a saved 1K checkpoint within the schedule')
    bundle = Path(record['bundle'])
    checkpoint = bundle/'model.pth' if step == 0 else root/f'train_model/{step//1000}epoch.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = root/f'evaluation_{step}'
    output.mkdir(exist_ok=False)
    (output/'audio').mkdir()
    model, config = load(bundle, checkpoint, 'cuda')
    model.eval().requires_grad_(False)
    torch.set_num_threads(4)
    panel = [json.loads(line) for line in (root/'panel.jsonl').read_text().splitlines()]
    rows = []
    for index, row in enumerate(panel, 1):
        ids = torch.tensor([config['token_list'].index(p) for p in row['expected_phones']], device='cuda')
        vector = kaldiio.load_mat(row['vector'])
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError('Invalid reference speaker vector')
        torch.manual_seed(777 + index)
        with torch.inference_mode():
            result = model.tts.inference(text=ids, spembs=torch.from_numpy(vector).to('cuda'),
                                         noise_scale=.667, noise_scale_dur=.8, max_len=5168)
        audio = result['wav'].detach().cpu().numpy().reshape(-1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Invalid synthesis')
        path = output/'audio'/f'{index:02d}.wav'
        sf.write(path, audio, 22050, subtype='FLOAT')
        for condition, audio_path in [('synthesis',path),('reference',Path(row['audio_path']))]:
            wav, rate = sf.read(audio_path,dtype='float32')
            rows.append(dict(id=row['id'], source_id=row['source_id'], condition=condition,
                             expected_phones=row['expected_phones'], audio_path=str(audio_path),
                             audio_sha256=sha256(audio_path), duration_seconds=len(wav)/rate,
                             clipping_fraction=float(np.mean(np.abs(wav)>=1.0)),
                             length_cap_reached=condition=='synthesis' and len(wav)>=5168*256))
    del model
    torch.cuda.empty_cache()
    manifest=output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    score(manifest,output/'phones.jsonl','cuda')
    torch.cuda.empty_cache()
    phones=[json.loads(line) for line in (output/'phones.jsonl').read_text().splitlines()]
    counts=summary(phones)
    quality=QualityModels(yaml.safe_load(Path('training/conf/quality.yaml').read_text())['models'])
    quality_rows=[]
    for row in rows:
        audio,rate=sf.read(row['audio_path'],dtype='float32')
        quality_rows.append(dict(id=row['id'],condition=row['condition'],**quality.score(audio,rate)))
    metrics={}
    for role in ['synthesis','reference']:
        for metric in ['sigmos_overall','sigmos_speech','sigmos_noise','sigmos_coloration','sigmos_discontinuity','audiobox_pq']:
            metrics[f'{role}_{metric}_median']=float(np.median([r[metric] for r in quality_rows if r['condition']==role]))
        for phone in ['r','l']:
            counter=counts[role][phone+'_counts']
            if sum(counter.values()):metrics[f'{role}_{phone}_exact_rate']=counter.get(phone,0)/sum(counter.values())
    references={r['id']:r for r in rows if r['condition']=='reference'}
    for row in rows:
        row['duration_ratio']=row['duration_seconds']/references[row['id']]['duration_seconds']
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary_row=dict(step=step,checkpoint_sha256=sha256(checkpoint),files=len(panel),
                     reference_policy='Fixed processed dev recordings resampled to 22050 Hz; filtering used scores at 24000 Hz',
                     conditioning='512D vector extracted from each reference recording; no reference durations or acoustic features passed',
                     phone_counts=counts,metrics=metrics,quality_model_identity=quality.identity)
    (output/'summary.json').write_text(json.dumps(summary_row,ensure_ascii=False,indent=2)+'\n')
    (output/'quality.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in quality_rows))
    with (output/'quality.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(quality_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(quality_rows)
    history=[]
    for path in sorted(root.glob('evaluation_*/summary.json'),key=lambda p:int(p.parent.name.split('_')[-1])):
        previous=json.loads(path.read_text())
        history.append(dict(step=previous['step'],**previous['metrics']))
    (root/'checkpoint_comparison.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    with (root/'checkpoint_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(history)
    events=root/('train_model/tensorboard/valid' if step else 'baseline_events/valid')
    with SummaryWriter(str(events)) as writer:
        for key,value in metrics.items():writer.add_scalar('listening/'+key,value,step)
    render(root, step)
    print('VITS_EVALUATION',json.dumps(summary_row,ensure_ascii=False),flush=True)


def render(root, step):
    """Rebuild the page from existing audio and metrics without GPU inference."""
    output = root/f'evaluation_{step}'
    summary_row = json.loads((output/'summary.json').read_text())
    if summary_row['step'] != step:
        raise ValueError('Evaluation summary step mismatch')
    metrics = summary_row['metrics']
    panel = [json.loads(line) for line in (root/'panel.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (output/'manifest.jsonl').read_text().splitlines()]
    quality_rows = [json.loads(line) for line in (output/'quality.jsonl').read_text().splitlines()]
    for row in rows:
        if sha256(Path(row['audio_path'])) != row['audio_sha256']:
            raise ValueError('Listening audio hash mismatch')
    parts=['<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           f'<title>Ukrainian VITS · {step:,}</title>',
           '<style>body{max-width:1100px;margin:auto;padding:24px;font:17px/1.5 system-ui}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}audio{width:100%}article{border-top:1px solid #aaa;margin-top:25px}small{display:block}@media(prefers-color-scheme:dark){body{background:#171717;color:#eee}}</style>',
           f'<h1>VITS · {step:,} кроків · декілька голосів</h1>',
           '<p>Фіксована validation-вибірка. Референс — оброблене аудіо 22,05 кГц; фільтр ≥3,5 застосовувався до версії 24 кГц. MOS і фонетичні мітки — автоматичні оцінки, не висновок слухача.</p>',
           '<pre>'+html.escape(json.dumps(metrics,indent=2))+'</pre>']
    by_audio={(r['id'],r['condition']):r for r in rows}
    by_quality={(r['id'],r['condition']):r for r in quality_rows}
    for row in panel:
        parts.append(f'<article><h2>{html.escape(row["text"])}</h2><p>{html.escape(row["source_id"])} · {html.escape(row["speaker_id"])}</p><div class="players">')
        for role,label in [('reference','Оброблений референс'),('synthesis',f'VITS · {step:,}')]:
            src=base64.b64encode(Path(by_audio[row['id'],role]['audio_path']).read_bytes()).decode()
            q=by_quality[row['id'],role]
            parts.append(f'<section><h3>{label}</h3><audio controls preload="none" src="data:audio/wav;base64,{src}"></audio><small>SigMOS {q["sigmos_overall"]:.3f} · Audiobox PQ {q["audiobox_pq"]:.3f}</small></section>')
        parts.append('</div></article>')
    parts.append('<script>document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>')
    (output/'preview.html').write_text('\n'.join(parts))
    print('VITS_LISTENING', str(output/'preview.html'), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--render-only',action='store_true',help='Rebuild HTML from existing evaluation files')
    args=parser.parse_args()
    (render if args.render_only else evaluate)(args.root,args.step)


if __name__=='__main__':main()
