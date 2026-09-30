from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    name: str | None
    family: str | None
    available_models: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.name is not None


def select_model(model_names: list[str] | tuple[str, ...]) -> ModelChoice:
    """Select Qwen first, then Llama, without assuming a particular model tag."""
    names = tuple(model_names)
    qwen = sorted((name for name in names if "qwen" in name.lower()), key=str.lower)
    llama = sorted((name for name in names if "llama" in name.lower()), key=str.lower)
    if qwen:
        return ModelChoice(qwen[0], "Qwen", names)
    if llama:
        return ModelChoice(llama[0], "Llama", names)
    return ModelChoice(None, None, names)
