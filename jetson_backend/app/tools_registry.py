"""
tools_registry.py — Extensible LLM tool registration.

To add a new capability to Baymax, call register_tool() once:

    register_tool(
        name="my_new_action",
        description="What the LLM should know about this tool and when to use it.",
        parameters={
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "What this param does"}
            },
            "required": ["param"],
        },
        handler=my_function,   # fn(**kwargs) -> str  (return a result string)
        tags=["actuator"],     # optional: group tools by subsystem
    )

The LLM discovers all registered tools automatically on every conversation turn.
No other files need changing.
"""

from __future__ import annotations
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# ── Registry storage ──────────────────────────────────────────────────────────
# Each entry:  { name, description, parameters (JSON Schema), handler, tags }
_REGISTRY: dict[str, dict] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict,
    handler: Callable,
    tags: list[str] | None = None,
) -> None:
    """Register a new LLM tool.  Overwrites silently if name already exists."""
    _REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "handler": handler,
        "tags": tags or [],
    }
    logger.info("Tool registered: %s  tags=%s", name, tags)


def get_gemini_tools():
    """Return tools as a Gemini SDK types.Tool object."""
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=entry["name"],
            description=entry["description"],
            parameters=entry["parameters"],
        )
        for entry in _REGISTRY.values()
    ]
    if not declarations:
        return None
    return [types.Tool(function_declarations=declarations)]


def dispatch(name: str, arguments: dict) -> str:
    """Call a registered tool handler and return its string result."""
    entry = _REGISTRY.get(name)
    if not entry:
        return f"Error: unknown tool '{name}'"
    try:
        result = entry["handler"](**arguments)
        return str(result) if result is not None else "done"
    except Exception as exc:
        logger.exception("Tool '%s' raised an exception", name)
        return f"Error running tool '{name}': {exc}"


def list_tools(tag: str | None = None) -> list[str]:
    """Return registered tool names, optionally filtered by tag."""
    if tag:
        return [n for n, e in _REGISTRY.items() if tag in e["tags"]]
    return list(_REGISTRY.keys())


# ── Built-in tools are registered lazily from llm_engine.py to avoid circular
#    imports.  Third-party extensions should import and call register_tool()
#    directly from their own module at startup.
