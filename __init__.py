"""memohood plugin entry point (DESIGN_v1.md "__init__.py: register(ctx):
ctx.register_memory_provider(MemoHoodMemoryProvider())").

Thin glue only, per this project's plugin convention (see
``plugins/disk-cleanup/__init__.py``): wires the
provider + CLI command into the ``ctx`` the hermes plugin loader hands us.
No logic lives here.

NOTE (see ``plugin.yaml``'s own inline comment): memory-provider plugins are
discovered by ``plugins/memory/__init__.py``'s own scanner under
``$HERMES_HOME/plugins/<name>/`` directly -- installing this package to
``$HERMES_HOME/plugins/memohood/`` (NOT ``$HERMES_HOME/plugins/memory/memohood/``) is
required for it to be found.
"""

from __future__ import annotations

from . import cli as memohood_cli
from .provider import MemoHoodMemoryProvider


def register(ctx) -> None:
    ctx.register_memory_provider(MemoHoodMemoryProvider())
    memohood_cli.register(ctx)
