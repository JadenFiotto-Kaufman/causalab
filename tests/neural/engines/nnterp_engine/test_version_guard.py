"""The version guard of a remote run (``nnterp_engine/versions.py``): before
any job is submitted, every package the server's code is made of
(``versions.GUARDED``) is this client's, by strict string equality — or the
run is refused (P4) naming the host, the package and both versions.

The server's ``/env`` is what nnsight caches per host, seeded here through
``nnsight.ndif.set_remote_env`` (``FaithfulServer.serve_env``); "no job" is
the faithful server's ``request`` never having been called.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from causalab.neural.engines.nnterp_engine import versions
from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.protocol.errors import ProtocolError

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnterp_engine.conftest import ROWS

pytestmark = pytest.mark.smoke

OTHER_HOST = "http://ndif-b.test:5001"


def _point(bundle, *, remote=None) -> NnterpExecutor:
    return sweep.make_executor(
        NnterpExecutor,
        sweep.interchange_doc("block_output", 1),
        bundle,
        rows=ROWS,
        with_cf=True,
        **({} if remote is None else {"remote": remote}),
    )


def test_a_matching_server_runs(remote_llama, ndif_llama):
    _point(remote_llama).run_all()
    assert len(ndif_llama.jobs) == 1


@pytest.mark.parametrize("package", versions.GUARDED)
@pytest.mark.parametrize("door", ["session", "lone_trace"])
def test_a_skewed_server_is_refused_before_any_job(
    remote_llama, ndif_llama, package, door
):
    """Every remote submission passes the guard — a point's session and a
    lazy read's lone trace alike — and a refusal costs no job."""
    ndif_llama.serve_env({package: "9.9.9"})
    executor = _point(remote_llama)
    with pytest.raises(ProtocolError) as refusal:
        executor.run_all() if door == "session" else executor.read_value("logits")
    assert refusal.value.code == "P4"
    message = str(refusal.value)
    here = importlib.metadata.version(package)
    assert "https://" in message or "http://" in message  # the host
    assert f"has {package} '9.9.9'" in message
    assert f"this client has {package} {here!r}" in message
    assert not ndif_llama.jobs


def test_a_server_without_the_package_is_refused(remote_llama, ndif_llama):
    ndif_llama.serve_env({"nnterp": None})
    with pytest.raises(ProtocolError, match="does not have 'nnterp' installed") as err:
        _point(remote_llama).run_all()
    assert err.value.code == "P4"
    assert not ndif_llama.jobs


def test_a_server_that_will_not_say_what_it_runs_is_refused(
    remote_llama, ndif_llama, monkeypatch
):
    """nnsight raises a plain ``RuntimeError`` when the ``/env`` fetch fails
    — an unreachable host, or one too old to serve it. Every other refusal on
    this path is a ``ProtocolError`` naming the host; so is this one, and it
    costs no job either."""
    import nnsight.ndif as ndif

    def unanswered(host: str) -> dict:
        raise RuntimeError("404 Not Found")

    monkeypatch.setattr(ndif, "get_remote_env", unanswered)
    with pytest.raises(ProtocolError, match="did not answer for its environment"):
        _point(remote_llama).run_all()
    assert not ndif_llama.jobs


def test_hosts_are_checked_independently(nnterp_llama, ndif_llama):
    """The configured host matches and another does not: a run against each
    gets its own host's verdict, whichever ran first."""
    ndif_llama.serve_env({"causalab": "0.0.0"}, host=OTHER_HOST)
    _point(nnterp_llama, remote=True).run_all()
    assert len(ndif_llama.jobs) == 1
    with pytest.raises(ProtocolError, match="ndif-b.test:5001 has causalab '0.0.0'"):
        _point(nnterp_llama, remote=OTHER_HOST).run_all()
    assert len(ndif_llama.jobs) == 1
    _point(nnterp_llama, remote=True).run_all()
    assert len(ndif_llama.jobs) == 2
    # a trailing slash is the same server (nnsight's `resolve_host`)
    with pytest.raises(ProtocolError, match="ndif-b.test:5001 has"):
        versions.ensure_server_matches(OTHER_HOST + "/")


def test_a_host_is_asked_once_per_process(remote_llama, ndif_llama, monkeypatch):
    import nnsight.ndif as ndif

    asked: list[str] = []
    real = ndif.get_remote_env

    def counting(host=None, **kwargs):
        asked.append(host)
        return real(host, **kwargs)

    monkeypatch.setattr(ndif, "get_remote_env", counting)
    _point(remote_llama).run_all()
    _point(remote_llama).run_all()
    assert len(asked) == 1 and len(ndif_llama.jobs) == 2


def test_no_server_is_asked_when_nothing_is_submitted(monkeypatch):
    """``remote=False`` submits nothing and ``"local"`` is nnsight's in-process
    dry run: neither has a server to ask."""
    import nnsight.ndif as ndif

    def unreachable(*_args, **_kwargs):
        raise AssertionError("the guard asked a server")

    monkeypatch.setattr(ndif, "get_remote_env", unreachable)
    versions.ensure_server_matches(False)
    versions.ensure_server_matches("local")
