import copy
import json

import pytest
import torch

from training.transfer.convert import initialize_rows
from training.transfer.vits import task, validate_config
from training.transfer.vits_run import run


def donor_config():
    return dict(tts='vits',feats_extract='linear_spectrogram',normalize=None,
                pitch_normalize=None,energy_normalize=None,pitch_extract=None,energy_extract=None,
                tts_conf={'sampling_rate':22050,'generator_params':{'spk_embed_dim':512}})


def test_reject_incompatible_normalization_and_speaker_encoder():
    config=donor_config();validate_config(config)
    for key in ['normalize','pitch_normalize','energy_normalize','pitch_extract','energy_extract']:
        changed=copy.deepcopy(config);changed[key]='global_mvn'
        with pytest.raises(ValueError):validate_config(changed)
    config['tts_conf']['generator_params']['spk_embed_dim']=192
    with pytest.raises(ValueError):validate_config(config)


def test_custom_task_parser_and_vits_metrics():
    from training.scripts.training_status import parse_log
    args=task().get_parser().parse_args(['--output_dir','exp','--token_list','tokens.txt'])
    assert args.embedding_lr_factor==10
    result=parse_log('2026-10-21 00:00:00,000 INFO: 2epoch:train:1-10batch: generator_mel_loss=32.5, generator_kl_loss=1.2, generator_dur_loss=2.3, train_time=0.9',1000)
    assert result['total_iterations']==1010
    assert result['latest_train_metrics']['generator_mel_loss']==32.5


def test_copied_r_is_independent_from_other_rows():
    donor=torch.randn(3,192)
    result=initialize_rows(donor,[{'target_id':0,'source_id':1},
                                 {'target_id':1,'source_id':2},
                                 {'target_id':2,'source_id':1}],777)
    assert torch.equal(result[0],donor[1]) and torch.equal(result[1],donor[2])
    result[0].add_(1)
    assert torch.equal(result[2],donor[1])


def test_50k_schedule_keeps_optimizer_resume_after_first_stage(tmp_path,monkeypatch):
    (tmp_path/'experiment.json').write_text(json.dumps({'evaluation_interval':5000,'steps':50000}))
    commands=[]
    monkeypatch.setattr('training.transfer.vits_run.subprocess.run',lambda cmd,check:commands.append(cmd))
    run(tmp_path)
    assert len(commands)==20
    for index in range(10):
        train,evaluate=commands[index*2:index*2+2]
        assert ('--resume' in train)==(index>0)
        assert train[train.index('--epochs')+1]==str((index+1)*5)
        assert evaluate[-1]==str((index+1)*5000)
