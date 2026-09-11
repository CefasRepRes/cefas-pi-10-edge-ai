try:
    from .tab import ApplicationValidationTab
except Exception:  # pragma: no cover - allows headless workflows to import submodules
    ApplicationValidationTab = None

__all__ = ["ApplicationValidationTab"]
