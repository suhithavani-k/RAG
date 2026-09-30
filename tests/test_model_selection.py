from core.model_selector import select_model


def test_qwen_has_priority_over_llama():
    choice = select_model(["llama3.2:3b", "qwen2.5:3b"])
    assert choice.name == "qwen2.5:3b"
    assert choice.family == "Qwen"
    assert choice.ready


def test_llama_is_fallback_when_qwen_is_absent():
    choice = select_model(["llama3.1:8b"])
    assert choice.name == "llama3.1:8b"
    assert choice.family == "Llama"


def test_no_supported_models_returns_unready_choice():
    choice = select_model(["nomic-embed-text:latest"])
    assert choice.name is None
    assert choice.family is None
    assert not choice.ready


def test_model_detection_is_case_insensitive_and_accepts_variants():
    assert select_model(["library/QwEn3:latest"]).name == "library/QwEn3:latest"
