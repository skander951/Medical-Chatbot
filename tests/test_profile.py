from medassist.utils.profile import merge_profile, missing_fields


def test_missing_fields_on_empty_profile():
    assert set(missing_fields({})) == {"age", "sex", "symptoms", "allergies", "current_medications"}


def test_merge_keeps_existing_values():
    existing = {"age": 29, "sex": None, "symptoms": None, "allergies": None, "current_medications": None}
    update = {"sex": "female", "age": None, "symptoms": None, "allergies": None, "current_medications": None}
    merged = merge_profile(existing, update)
    assert merged["age"] == 29
    assert merged["sex"] == "female"


def test_merge_accumulates_symptoms():
    existing = {"symptoms": "fever"}
    update = {"symptoms": "dry cough"}
    merged = merge_profile(existing, update)
    assert "fever" in merged["symptoms"] and "dry cough" in merged["symptoms"]


def test_none_and_placeholder_values_ignored():
    merged = merge_profile({"allergies": "none"}, {"allergies": "unknown"})
    assert merged["allergies"] == "none"
