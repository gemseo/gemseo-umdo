from __future__ import annotations

import ast
import inspect
from typing import Any

from griffe import Attribute
from griffe import Class
from griffe import Docstring
from griffe import dynamic_import
from griffe import Extension
from griffe import get_logger
from griffe import Object
from griffe import ObjectNode

logger = get_logger(__name__)


class DocstringInheritance(Extension):
    """Inherit docstrings when the package docstring-inheritance is used."""

    def on_class_members(self, node: ast.AST | ObjectNode, cls: Class) -> None:
        if isinstance(node, ObjectNode):
            return  # skip runtime objects, their docstrings are already right

        runtime_cls = self.__import_dynamically(cls)

        if not self.__has_docstring_inheritance(runtime_cls):
            return

        self.__set_docstring(cls, runtime_cls)

        for member in cls.members.values():
            if isinstance(member, Attribute):
                for cls_ in reversed(runtime_cls.mro()):
                    runtime_member = getattr(cls_, member.name, None)
                    if runtime_member is not None:
                        self.__set_docstring(member, runtime_member)
                        break
            else:
                runtime_obj = self.__import_dynamically(member)
                self.__set_docstring(member, runtime_obj)

    @staticmethod
    def __import_dynamically(obj: Object) -> Any:
        """Import dynamically and return an object."""
        try:
            return dynamic_import(obj.path)
        except ImportError:
            logger.debug("Could not get dynamic docstring for %s", obj.path)

    @staticmethod
    def __set_docstring(obj: Object, runtime_obj: Any) -> None:
        """Set the docstring from a runtime object.

        Args:
            obj: The griffe object.
            runtime_obj: The runtime object.
        """
        if runtime_obj is None:
            return

        try:
            docstring = runtime_obj.__doc__
        except AttributeError:
            logger.debug("Object %s does not have a __doc__ attribute", obj.path)
            return

        if docstring is None:
            return

        # update the object instance with the evaluated docstring
        docstring = inspect.cleandoc(docstring)
        if obj.docstring:
            obj.docstring.value = docstring
        else:
            obj.docstring = Docstring(docstring, parent=obj)

    def __has_docstring_inheritance(self, cls: Any) -> bool:
        for base in cls.__class__.__mro__:
            if base.__name__.endswith("DocstringInheritanceMeta"):
                return True
        return False
