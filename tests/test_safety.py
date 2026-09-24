from medassist.utils.safety import is_emergency


def test_detects_chest_pain():
    assert is_emergency("I have severe chest pain and can't breathe")


def test_detects_self_harm():
    assert is_emergency("I want to kill myself")


def test_normal_message_is_not_emergency():
    assert not is_emergency("I have a mild headache since this morning")


def test_empty_string_is_safe():
    assert not is_emergency("")
