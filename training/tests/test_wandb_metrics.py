import math

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from training.quality.wandb_metrics import collect_scalars


def write_events(path, events):
    writer = EventFileWriter(str(path))
    for event in events:
        writer.add_event(event)
    writer.close()


def scalar(tag, value, step, wall_time):
    return Event(step=step, wall_time=wall_time,
                 summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]))


def test_scalar_export_preserves_steps_and_skips_media_and_nonfinite(tmp_path):
    write_events(tmp_path / 'train', [
        scalar('loss', 4, 10, 1), scalar('loss', 3, 20, 2),
        scalar('bad', math.nan, 20, 2),
        Event(step=20, wall_time=2, summary=Summary(value=[
            Summary.Value(tag='audio', audio=Summary.Audio(encoded_audio_string=b'private')),
        ])),
    ])
    write_events(tmp_path / 'valid', [scalar('loss', 5, 20, 3)])
    rows, cursors = collect_scalars(tmp_path, {})
    assert rows == [{'training_step': 10, 'train/loss': 4},
                    {'training_step': 20, 'train/loss': 3, 'valid/loss': 5}]
    assert collect_scalars(tmp_path, cursors)[0] == []
    # Epoch aggregates can update the same tag at the same optimizer step.
    write_events(tmp_path / 'train', [scalar('loss', 2.5, 20, 4)])
    assert collect_scalars(tmp_path, cursors)[0] == [
        {'training_step': 20, 'train/loss': 2.5}]


def test_missing_event_directories_are_safe_before_training(tmp_path):
    assert collect_scalars(tmp_path, {}) == ([], {})
