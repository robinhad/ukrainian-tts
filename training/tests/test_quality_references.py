import io
import struct
import wave

import numpy as np

from training.quality.references import decode_native


def test_invalid_utf8_container_tag_does_not_change_decoded_audio():
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.arange(-100, 100, dtype='<i2').tobytes())
    original = stream.getvalue()
    # RIFF INFO title containing a byte that cannot be decoded as UTF-8.
    info = b'INFO' + b'INAM' + struct.pack('<I', 2) + b'\xa7\x00'
    tagged = original + b'LIST' + struct.pack('<I', len(info)) + info
    tagged = tagged[:4] + struct.pack('<I', len(tagged) - 8) + tagged[8:]
    expected, rate = decode_native(original)
    actual, tagged_rate = decode_native(tagged)
    assert rate == tagged_rate == 16000
    np.testing.assert_array_equal(actual, expected)
