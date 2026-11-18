"""Ładowanie z HF: najpierw lokalny cache (bez sieci), Hub tylko gdy cache nie ma."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from _utils_.log_print import log_print

T = TypeVar("T")


@contextmanager
def _hf_offline() -> Iterator[None]:
    import datasets.config as ds_config
    import huggingface_hub.constants as hub_constants

    prev_ds = ds_config.HF_HUB_OFFLINE
    prev_hub = hub_constants.HF_HUB_OFFLINE
    ds_config.HF_HUB_OFFLINE = True
    hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        ds_config.HF_HUB_OFFLINE = prev_ds
        hub_constants.HF_HUB_OFFLINE = prev_hub


def offline_first(what: str, load: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        with _hf_offline():
            return load(*args, **kwargs)
    except Exception as e:
        log_print(f"{what}: brak w lokalnym cache HF ({type(e).__name__}) — pobieranie z Hub")
    return load(*args, **kwargs)
