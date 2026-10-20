"""Real checkpoint/tokenization/GAN backward smoke test; no optimizer steps."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .convert import EMBEDDING, load_strict
from .frontend import TransferFrontend
from .prepare import prepare, read_audio
from .train import training_config


def smoke(bundle, pairs, device, output, flash_attention=False, checkpoint=None):
    from espnet2.tasks.gan_tts import GANTTSTask
    from espnet2.train.distributed_utils import DistributedOption
    from espnet2.torch_utils.load_pretrained_model import load_pretrained_model
    torch.set_num_threads(4)
    torch.manual_seed(777)
    checkpoint = checkpoint or bundle / 'model.pth'
    model, args = load_strict(bundle, device, checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert set(model.state_dict()) == set(state)
    assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
    # Exercise the actual trainer's init_param loader, not only strict inference.
    load_pretrained_model(str(checkpoint), model, ignore_init_mismatch=False)
    assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
    frontend = TransferFrontend(args.token_list)
    frontend.verify(bundle / 'frontend.json')
    assert 'ˈa' in frontend.phonemes('за́мок')
    assert 'ˈo' in frontend.phonemes('замо́к')
    pair = json.loads(pairs.read_text().splitlines()[0])
    phones, ids = frontend.encode(pair['text'])
    # Exact IDs through the training preprocessor versus inference frontend.
    preprocessor = GANTTSTask.build_preprocess_fn(args, train=True)
    training_ids = preprocessor('smoke', {'text': ' '.join(phones)})['text']
    assert np.array_equal(training_ids, np.asarray(ids))
    audio = read_audio(pair['audio_path'], args.feats_extract_conf['fs'])
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        one_pair = temporary / 'pair.jsonl'
        one_pair.write_text(json.dumps(pair) + '\n')
        prepared = temporary / 'prepared'
        prepare(bundle, one_pair, prepared)
        data_args = argparse.Namespace(**training_config(bundle, prepared, prepared, temporary / 'run'))
        data_args.num_workers = 0
        data_args.num_iters_per_epoch = 1
        iterator = GANTTSTask.build_iter_factory(data_args, DistributedOption(), 'train')
        _, batch = next(iter(iterator.build_iter(1)))
        assert batch['text'][0].tolist() == ids
        assert batch['speech_lengths'].tolist() == [len(audio)]
        batch = {key: value.to(device) for key, value in batch.items()}
    optimizers = GANTTSTask.build_optimizers(args, model)
    assert all(not optim.state for optim in optimizers)
    assert all(p.requires_grad for p in model.parameters())
    optimized = {id(p) for optim in optimizers for group in optim.param_groups for p in group['params']}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    model.train()
    if flash_attention:
        from .attention import enable_flash_attention
        flash_modules = enable_flash_attention(model)
    results = {}
    for name, generator in [('generator', True), ('discriminator', False)]:
        model.zero_grad(set_to_none=True)
        loss = model(**batch, forward_generator=generator)['loss']
        assert torch.isfinite(loss).all()
        loss.backward()
        gradients = {key: param.grad for key, param in model.named_parameters() if param.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        prefix = 'tts.generator.' if generator else 'tts.discriminator.'
        assert any(g.abs().sum() > 0 for k, g in gradients.items() if k.startswith(prefix) and k != EMBEDDING)
        if generator:
            assert gradients[EMBEDDING].abs().sum() > 0
        results[name] = dict(loss=float(loss.detach()), gradient_tensors=len(gradients))
    model.eval()
    with torch.no_grad():
        wav = model.tts.inference(text=batch['text'][0])['wav']
    assert wav.numel() > 0 and torch.isfinite(wav).all()
    # No optimizer.step(): all parameter tensors remain unchanged. Spectral-norm
    # running buffers, if present in other donors, are not parameter tensors.
    assert all(torch.equal(param.detach().cpu(), state[key]) for key, param in model.named_parameters())
    results.update(strict_load=True, trainer_init_load=True, frontend_ids_identical=True,
                   prepared_data_iterator=True,
                   all_parameters_trainable=True, fresh_optimizers=True, optimizer_steps=0,
                   explicit_lexical_stress=True, inference_samples=wav.numel(),
                   sample_rate=args.feats_extract_conf['fs'], input_phones=len(ids),
                   input_samples=len(audio), learning_rate=args.optim_conf['lr'])
    if flash_attention:
        counts = {name: module.flash_calls for name, module in model.named_modules()
                  if hasattr(module, 'flash_calls')}
        assert len(counts) == len(flash_modules) and all(count > 0 for count in counts.values())
        results['flash_attention_calls'] = counts
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--flash-attention', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='Smoke a trained weights-only checkpoint')
    args = parser.parse_args()
    smoke(args.bundle, args.pairs, args.device, args.output, args.flash_attention, args.checkpoint)


if __name__ == '__main__':
    main()
