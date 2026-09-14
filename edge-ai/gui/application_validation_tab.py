"""Compatibility shim.

Keeps the existing app.py import working:
    from application_validation_tab import ApplicationValidationTab
"""
from application_validation.tab import ApplicationValidationTab

__all__ = ["ApplicationValidationTab"]
