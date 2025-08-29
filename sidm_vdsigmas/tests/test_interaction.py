import pytest

from sidm_vdsigmas.interaction import Interaction


def test_empty_interaction():
    with pytest.raises(ValueError):
        Interaction()
