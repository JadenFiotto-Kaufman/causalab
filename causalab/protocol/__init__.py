"""The engine-free intervention-protocol layer.

``docs/intervention_protocol.md`` is the normative spec; this package is
its compiler (:func:`compile_protocol`, the one path every entry point
takes), loader, validator, canonicalizer, sweep expander, planner, engine
contract, run entry point, and CLI. Nothing here imports torch or an execution
engine — a document is a value, and this layer owns everything decidable from
the value plus a resolution environment. :func:`run_protocol` is the exception
that proves the rule: it *executes* a document, and does so by taking the
engines as an argument rather than importing one.
"""

from causalab.protocol.engine import (
    Engine,
    ExecutionRequest,
    RunResult,
    choose_engine,
    requires,
)
from causalab.protocol.canonical import canonical_bytes, canonicalize, digest
from causalab.protocol.compile import CompiledProtocol, compile_protocol
from causalab.protocol.errors import (
    ParseError,
    ProtocolError,
    ValidationError,
    ValidationErrors,
)
from causalab.protocol.loader import LoadedProtocol, load
from causalab.protocol.plan import PointPlan, plan_point
from causalab.protocol.resolution import (
    Available,
    Denominator,
    Invalid,
    Resolution,
    Unavailable,
)
from causalab.protocol.run import RUN_RECORD_NAME, run_protocol
from causalab.protocol.resolve import (
    ArtifactStore,
    DatasetResolver,
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
)
from causalab.protocol.schema import Document, load_raw, parse_document
from causalab.protocol.sweep import Expansion, expand, find_axes
from causalab.protocol.validate import validate_document

__all__ = [
    "RUN_RECORD_NAME",
    "ArtifactStore",
    "Available",
    "CompiledProtocol",
    "Denominator",
    "Engine",
    "DatasetResolver",
    "Document",
    "ExecutionRequest",
    "Expansion",
    "FileArtifacts",
    "FileDatasets",
    "Invalid",
    "LoadedProtocol",
    "ParseError",
    "PointPlan",
    "ProtocolError",
    "Resolution",
    "ResolutionEnv",
    "RunResult",
    "Unavailable",
    "ValidationError",
    "ValidationErrors",
    "canonical_bytes",
    "canonicalize",
    "choose_engine",
    "compile_protocol",
    "digest",
    "expand",
    "find_axes",
    "load",
    "load_raw",
    "parse_document",
    "plan_point",
    "requires",
    "run_protocol",
    "validate_document",
]
