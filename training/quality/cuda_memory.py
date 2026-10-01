"""Release unused allocator cache between complete GAN updates, with evidence."""
from __future__ import annotations

import json
import logging


def release_cuda_cache(step, cuda=None):
    if cuda is None:
        from torch import cuda

    def snapshot():
        return {
            "allocated_gib": cuda.memory_allocated() / 2**30,
            "reserved_gib": cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": cuda.max_memory_reserved() / 2**30,
        }

    before = snapshot()
    cuda.empty_cache()
    record = {"epoch_step": step, "before": before, "after": snapshot()}
    logging.info("cuda_cache_release=%s", json.dumps(record, sort_keys=True))
    return record
