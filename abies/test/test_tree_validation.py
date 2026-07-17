from decimal import Decimal

from apps.campionamenti.tree_validation import normalize_sample_tree_values


def test_sample_tree_values_default_h_measured_false():
    values = normalize_sample_tree_values(
        number=1, d_cm=30, h_m=Decimal('20.123'),
        shoot=0, l10_mm=10, pressler_coeff=Decimal('2.345'),
    )

    assert values.h_measured is False
    assert values.h_m == Decimal('20.12')
    assert values.pressler_coeff == Decimal('2.35')


def test_sample_tree_values_preserve_h_measured_true():
    values = normalize_sample_tree_values(
        number=1, d_cm=30, h_m=Decimal('20.00'), h_measured=True,
    )

    assert values.h_measured is True


def test_sample_tree_values_reject_invalid_measurements():
    assert normalize_sample_tree_values(
        number=0, d_cm=30, h_m=Decimal('20.00'),
    ) is None
    assert normalize_sample_tree_values(
        number=1, d_cm=0, h_m=Decimal('20.00'),
    ) is None
    assert normalize_sample_tree_values(
        number=1, d_cm=30, h_m=Decimal('0'),
    ) is None
    assert normalize_sample_tree_values(
        number=1, d_cm=30, h_m=Decimal('20.00'), pressler_coeff=Decimal('0'),
    ) is None
