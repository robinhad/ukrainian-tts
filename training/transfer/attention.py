"""Opt-in FlashAttention-2 for the donor's plain JETS self-attention."""
import logging
from types import MethodType

import torch


def flash_forward(self, query, key, value, mask, expand_kv=False):
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    if query.device.type != 'cuda':
        raise ValueError('FlashAttention requires CUDA; use the default backend on CPU')
    if expand_kv or self.causal or self.cross_attn:
        raise ValueError('This adapter supports noncausal JETS self-attention only')
    if mask is not None and (mask.ndim != 3 or mask.shape[1] != 1):
        raise ValueError('Expected the JETS key padding mask [batch, 1, keys]')
    q, k, v = self.forward_qkv(query, key, value)
    output_dtype = q.dtype
    # Keep projections, parameters, optimizer and GAN losses in FP32. Only the
    # fused attention operation uses BF16, with FP32 internal accumulation.
    q, k, v = [x.transpose(1, 2).to(torch.bfloat16).contiguous() for x in (q, k, v)]
    dropout = self.dropout_rate if self.training else 0.0
    if mask is None:
        out = flash_attn_func(q, k, v, dropout_p=dropout, causal=False)
    else:
        batch, query_length = q.shape[:2]
        keep = mask[:, 0].bool()
        lengths = keep.sum(-1, dtype=torch.int32)
        if not torch.all(lengths > 0):
            raise ValueError('Every item must have at least one nonpadding key')
        indices = keep.reshape(-1).nonzero(as_tuple=False).flatten()
        cu_k = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
        cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * query_length
        # Preserve ALL queries, including padding: ESPnet masks keys only, and
        # later convolution layers can read padded query positions.
        out = flash_attn_varlen_func(
            q.flatten(0, 1), k.flatten(0, 1)[indices], v.flatten(0, 1)[indices],
            cu_q, cu_k, query_length, k.shape[1], dropout_p=dropout, causal=False,
        ).reshape(batch, query_length, self.h, self.d_k)
    self.flash_calls += 1
    if self.flash_calls == 1:
        logging.info('FlashAttention active: %s, BF16 attention, dropout=%s',
                     self.flash_name, dropout)
    return self.linear_out(out.flatten(2).to(output_dtype))


def enable_flash_attention(model):
    """Replace computation only; keep every parameter and state-dict key."""
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
    from espnet2.legacy.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, MultiHeadedAttention)]
    if not modules or any(type(module) is not MultiHeadedAttention for _, module in modules):
        raise ValueError('Flash adapter requires plain transformer attention')
    for name, module in modules:
        module.forward = MethodType(flash_forward, module)
        module.flash_calls = 0
        module.flash_name = name
    logging.info('Enabled FlashAttention-2 in %d attention modules', len(modules))
    return [name for name, _ in modules]


def flash_task():
    """Task subclass records the backend in ESPnet's saved training config."""
    from espnet2.tasks.gan_tts import GANTTSTask

    class FlashGANTTSTask(GANTTSTask):
        @classmethod
        def add_task_arguments(cls, parser):
            super().add_task_arguments(parser)
            parser.add_argument('--transfer_attention_backend', choices=['flash2'], default='flash2')

        @classmethod
        def build_model(cls, args):
            model = super().build_model(args)
            enable_flash_attention(model)
            return model

    return FlashGANTTSTask
