# core/models/__init__.py
from core.models.registry import ModelRegistry
from core.models.base_model import BaseModel

__all__ = ['ModelRegistry', 'BaseModel']