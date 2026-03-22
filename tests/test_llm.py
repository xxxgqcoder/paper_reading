from paper_reading.llm import resolve_model_name


def test_resolve_model_name_maps_openrouter_qwen_flash_alias() -> None:
    resolved = resolve_model_name(
        "qwen3.5-flash",
        endpoint="https://openrouter.ai/api/v1",
    )

    assert resolved == "qwen/qwen3.5-flash-02-23"


def test_resolve_model_name_maps_openrouter_qwen_vl_alias() -> None:
    resolved = resolve_model_name(
        "qwen2.5vl:7b",
        endpoint="https://openrouter.ai/api/v1",
    )

    assert resolved == "qwen/qwen-2.5-vl-7b-instruct"


def test_resolve_model_name_keeps_local_model_name_for_ollama() -> None:
    resolved = resolve_model_name(
        "qwen3.5-flash",
        endpoint="http://127.0.0.1:11434",
    )

    assert resolved == "qwen3.5-flash"
