"""Conservative boundary-only trimming from MFA's phone alignment JSON."""
import math


SILENCE_PHONES = {'', 'sil', 'sp', '<eps>'}
UNKNOWN_PHONES = {'spn', '<unk>', 'unk', 'oov'}


def boundaries(alignment, duration, padding_seconds=.1, max_removed_fraction=.5):
    """Return a proposal and review flags; callers preserve full audio on flags."""
    if duration <= 0 or padding_seconds < 0 or not 0 < max_removed_fraction < 1:
        raise ValueError('Invalid trimming configuration')
    tiers = alignment.get('tiers', {})
    phone_tiers = [v for k, v in tiers.items() if k == 'phones' or k.endswith(' - phones')]
    if len(phone_tiers) != 1:
        return 0., duration, ['missing_or_multiple_phone_tiers']
    speech, flags = [], []
    previous_end = 0.
    for start, end, phone in phone_tiers[0].get('entries', []):
        if (not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start
                or start < previous_end - .001 or end > duration + .03):
            return 0., duration, ['invalid_phone_intervals']
        previous_end = end
        # MFA may export Kaldi's positional suffixes.
        phone = phone.rsplit('_', 1)[0] if phone.endswith(('_B', '_E', '_I', '_S')) else phone
        if phone in UNKNOWN_PHONES:
            flags.append('unknown_phone')
        if phone not in SILENCE_PHONES:
            speech.append((start, end))
    if not speech:
        return 0., duration, ['no_aligned_speech']
    start = max(0., speech[0][0] - padding_seconds)
    end = min(duration, speech[-1][1] + padding_seconds)
    if end - start < .5:
        flags.append('short_speech_span')
    if 1 - (end - start) / duration > max_removed_fraction:
        flags.append('large_boundary_removal')
    return start, end, sorted(set(flags))
