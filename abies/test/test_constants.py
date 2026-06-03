"""Tests for `config.constants.is_truthy` — the unified truthy-token parser.

One parser spans both edges: form/JSON values (the HTML checkbox 'on', real
booleans) and CSV cells (Italian sì).  It is case-insensitive and
whitespace-trimmed; every other token is falsy.
"""

import pytest

from config.constants import is_truthy


class TestIsTruthy:
    @pytest.mark.parametrize('value', [
        'true', 'True', 'TRUE', '  true  ',   # case-insensitive, trimmed
        '1', 'yes', 'YES',
        'si', 'sì', 'SÌ',                      # Italian — the CSV edge
        'on',                                  # HTML checkbox — the form edge
        True, 1,                               # real bool / int stringify in
    ])
    def test_truthy_tokens(self, value):
        assert is_truthy(value) is True

    @pytest.mark.parametrize('value', [
        'false', 'False', '0', 'no', 'off', '', '   ', 'abc',
        None, False, 0,
    ])
    def test_everything_else_is_falsy(self, value):
        assert is_truthy(value) is False
