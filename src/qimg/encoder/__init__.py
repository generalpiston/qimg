from qimg.encoder.device import resolve_device

__all__ = ["LocalImageEncoder", "LocalTextEncoder", "resolve_device"]


def __getattr__(name: str):
    if name in {"LocalImageEncoder", "LocalTextEncoder"}:
        from qimg.encoder.encoder import LocalImageEncoder, LocalTextEncoder

        return {"LocalImageEncoder": LocalImageEncoder, "LocalTextEncoder": LocalTextEncoder}[name]
    raise AttributeError(name)
