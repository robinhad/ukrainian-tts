from training.scripts.plan_enhancement_workers import plan_assignments


def test_plan_assignments_balances_skewed_pending_shards() -> None:
    total_records = 1_440
    num_shards = 144
    completed = {shard: 10 for shard in range(num_shards)}
    for shard, done in {
        0: 0,
        17: 1,
        35: 2,
        52: 3,
        78: 4,
        101: 5,
        130: 6,
    }.items():
        completed[shard] = done

    assignments, loads = plan_assignments(
        total_records,
        completed,
        num_shards,
        worker_count=4,
    )

    assert sorted(shard for slot in assignments for shard in slot) == [
        0,
        17,
        35,
        52,
        78,
        101,
        130,
    ]
    assert max(loads) - min(loads) <= 4
    assert sum(loads) == 49
