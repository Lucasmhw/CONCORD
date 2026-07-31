from concord.engine import _accumulation_divisor


def test_partial_gradient_accumulation_group_uses_its_actual_size() -> None:
    divisors = [_accumulation_divisor(index, total_batches=10, accumulation_steps=4) for index in range(10)]
    assert divisors == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
