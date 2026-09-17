"""The version guard of a remote run: the server runs the code this client
planned for.

A block of this engine is ``causalab``'s own module-level functions (and
``nnterp``'s accessors), imported **where the block runs**, reading a program
planned **here**. A skew between the two installs is a skew between the plan
and the code that reads it — a renamed field, a changed default — and nothing
downstream would name it. So before any job is submitted the server's
environment is looked up (``nnsight.ndif.get_remote_env``, which nnsight
caches per host) and the two installs are compared by strict string equality,
package by package.

Once per ``(process, host)``: a host that matched is remembered here and not
asked again. ``remote="local"`` — nnsight's in-process dry run — has no
server, and ``remote=False`` no submission; neither is checked.
"""

from __future__ import annotations

import importlib.metadata

from causalab.protocol.errors import ProtocolError

__all__ = ["GUARDED", "ensure_server_matches"]

#: The packages whose code a block imports where it runs, by import name —
#: how a server's ``/env`` keys its packages.
GUARDED = ("causalab", "nnterp")

#: The resolved hosts whose environment matched, this process.
_MATCHED: set[str] = set()


def ensure_server_matches(remote: bool | str) -> None:
    """Refuse (P4) a remote submission to a server whose ``causalab`` or
    ``nnterp`` is not this client's, or is absent; nothing is submitted
    first. ``remote`` is what the run method takes: ``True`` for the
    configured host, a host URL, or ``False`` / ``"local"`` (no server)."""
    if not remote or remote == "local":
        return
    import nnsight.ndif as ndif

    host = ndif.resolve_host(remote if isinstance(remote, str) else None)
    if host in _MATCHED:
        return
    served = ndif.get_remote_env(host)["packages"]
    for name in GUARDED:
        here = importlib.metadata.version(name)
        there = served.get(name)
        if there == here:
            continue
        found = (
            f"does not have {name!r} installed (its /env lists no such package)"
            if there is None
            else f"has {name} {there!r}"
        )
        raise ProtocolError(
            "P4",
            f"remote run refused before any job was submitted: the NDIF server "
            f"at {host} {found}; this client has {name} {here!r}. The block is "
            f"{name}'s own functions imported where it runs, reading a program "
            "planned here, so the two installs must be the same version. "
            f"Install {name}=={here} on the server, install the server's "
            "version here, or point `remote` at a matching deployment.",
        )
    _MATCHED.add(host)
