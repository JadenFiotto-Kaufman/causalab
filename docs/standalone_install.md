# Standalone install — hash-locked, resolver-free

For installing causalab into an environment you do not control the resolution
of: a cluster job image, a colleague's venv, a reviewer reproducing a number.
`uv.lock` already pins every version, but only `uv` reads it, and letting a
resolver choose is precisely what a reproduction cannot afford.

`requirements.lock.txt` is that file: every dependency pinned with `==`, every
artifact named by its SHA-256, nothing left to resolve. It is generated from
`uv.lock` (a `uv export`, see [Regenerating](#regenerating)) and committed, and
CI fails on a diff, exactly as it does for `uv.lock`.

The recipe has two halves with two audiences. The **producer** has a checkout
and `uv`, and hands over two files. The **consumer** has those two files and a
Python — no `uv`, no checkout, no resolver — and that is the whole point.

## Producer: from a checkout

```bash
uv build --wheel          # -> dist/causalab-<version>-cp310-abi3-<platform>.whl
```

The wheel carries a compiled extension — the weight reader's Rust core,
`causalab.io.fastersafetensors._core` (`docs/fastersafetensors.md`) — so the
build needs a Rust toolchain (`rustup` on `PATH`; `rust-toolchain.toml` pins the version) and the wheel is
specific to the producer's platform (one `abi3` wheel serves every CPython
≥ 3.10 there). Build on the platform the consumer runs; the consumer needs no
toolchain — that is what shipping a wheel buys.

Hand over `dist/causalab-*.whl` **and** `requirements.lock.txt` — a release
artifact, an internal index, a shared path. The lock is committed, so the two
files describe the same tree whenever they come from the same commit.

## Consumer: from the wheel and the lock

```bash
# 1. the dependency closure, hash-verified. --require-hashes makes pip refuse
#    any artifact whose bytes do not match, and refuse the whole file if a
#    single requirement lacks a hash.
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --require-hashes -r requirements.lock.txt

# 2. causalab itself, from the wheel you were handed, with --no-deps. The
#    closure is already installed and exact; letting pip re-read the wheel's
#    metadata is the resolution step this file exists to avoid.
/path/to/venv/bin/pip install --no-deps causalab-*.whl

# 3. prove the install is one consistent set and the CLI is on PATH.
/path/to/venv/bin/pip check
/path/to/venv/bin/causalab --help
```

That is the resolver-free install, complete. The shipped documents are inside
the wheel — `causalab/configs/protocols/` — so
a consumer can address them without a checkout:

```bash
/path/to/venv/bin/python -c 'import causalab, pathlib; print(pathlib.Path(causalab.__file__).parent / "configs/protocols")'
```

What is **not** in the wheel is data. `validate` and `run` both read the rows a
document names (`weekdays/train`, for the shipped ones) from `--data-root`, and
the fixture rows live in the repository's `tests/protocol/fixtures/data`. A
consumer who wants to go beyond step 3 supplies a data root of their own, or
uses a checkout — which is what the smoke check below does.

## The smoke check: from a checkout

After the producer step and the three consumer steps, two more steps need the
checkout's fixture rows:

```bash
# 4. the pure layer, without a GPU or a downloaded checkpoint. `validate`
#    decides the whole §5 load-error checklist from the static model registry,
#    so this answers "is my install working" before any weights exist.
#    $CONFIGS is the installed configs/ directory printed above.
/path/to/venv/bin/causalab validate "$CONFIGS/protocols/interchange.json" \
  --data-root tests/protocol/fixtures/data

# 5. a real forward pass, a real write and two real metrics, on CPU.
#    `minimal_cpu.json` applies the shipped interchange method to a tiny random
#    Llama pinned to a commit SHA, so it downloads megabytes rather than
#    gigabytes and cannot be moved by an upstream re-upload. It writes iia.json
#    (a match accuracy) and logit_diff.json (a logit difference).
/path/to/venv/bin/causalab run "$CONFIGS/protocols/minimal_cpu.json" \
  --data-root tests/protocol/fixtures/data \
  --artifacts-root tests/protocol/fixtures/artifacts \
  --out /tmp/standalone-run
```

The documents are addressed *in the wheel*, not in the checkout, so the smoke
check also proves the packaged copy is the one that runs. It is worth running
whenever the packaging surface changes — `pyproject.toml`, the lock, the build
backend — since a break there is red before merge rather than after.

## Why causalab is not in the lock

The export passes `--no-emit-project`. Emitting the project would write a
**local path** — the one thing a standalone consumer cannot install from — so
the file describes the closure and step 2 installs the package. That split is
what lets one committed file serve a wheel install, an sdist install and a
cluster image alike.

## Why the extras are not in it

The base lock excludes all optional extras:

- **`notebook`** is a jupyter-server and Dash stack. A headless install has no
  use for it; a headless GPU job image that merged causalab into a
  network-less venv paid for all of it, which is why the extra exists at all.
- **`nnsight`** is a **git** dependency, pinned to a verified revision. A git
  revision has no artifact hash, and `--require-hashes` refuses a file
  containing even one unhashable requirement — so including it would disable
  hash-locking for *every* consumer, not just the ones who wanted nnsight.
- **`flash-attn` / `flash-linear-attention`** supply optional Linux GPU
  kernels. The base install keeps Transformers' fallbacks; installation and
  selection are described in [attention backends](attention_backends.md).

Extras are installable the ordinary way on top of a hash-locked base — from the
same wheel, naming the extra (causalab is unpublished, so a bare
`causalab[notebook]` has no index to resolve against):

```bash
/path/to/venv/bin/pip install 'causalab-*.whl[notebook]'
```

**This step resolves, and resolution can move a package the closure pinned.**
The jupyter and Dash stacks share transitive dependencies with the core set
(`tornado`, `traitlets`, `jinja2` and their neighbours), and `pip` will upgrade
a shared one if the extra asks. An environment that has taken this step is no
longer the one `requirements.lock.txt` describes, and nothing tells you so —
diff `pip freeze` against the lock afterwards, or keep the reproduction venv
and the interactive venv separate. (A `-c requirements.lock.txt` constraint
would be the clean answer, but `pip` rejects a constraints file that carries
`--hash` lines.)

The lock is checked so that nothing unhashable ever reaches it, and this stays
true by check rather than by memory.

## The lock is platform-independent

`uv export` emits the lockfile's whole marker set rather than resolving for the
current machine, so the linux-only `nvidia-cu12` wheels and the win32-only
packages are present *with their markers* even when the export ran on macOS. The
file is therefore byte-identical whatever regenerates it — which is what makes
the CI diff a check rather than a guaranteed failure on a different runner.

## The smoke check, continued: every workflow document

The fuller check drives `validate --data` and `explain`, with the installed
`causalab` executable, over every workflow document in `demos/` and
`causalab/configs/workflows/`, from a temporary working directory that is
neither the checkout nor a document's own. A workflow's `{"module": …}`
script locators are resolved by importing the named modules' parent packages,
so this is where a dependency the dev group carries and the wheel does not
first fails, and where a `path` that leaned on a repository root would. Run it
from a process that never imports `causalab` itself, so it exercises the
install under test and nothing else.

## Regenerating

The file is `uv.lock` exported with hashes and nothing left to resolve:

```bash
uv export --frozen --no-dev --no-emit-project --format requirements-txt --hashes
```

under the short header the committed file carries. `--frozen` is load-bearing:
without it the export could re-resolve, and the file would no longer be a
function of the committed lock. Any change to `pyproject.toml`'s core
dependencies, or to `uv.lock`, moves this file. Regenerate it in the **same
commit** — a stale lock is a red PR, and a red `main` if it slips through. On a merge conflict in the file, regenerate;
never merge it by hand — the header's "do not hand-edit" applies to conflict
markers too.

**The file is a byte-for-byte function of the `uv` that wrote it.** The
export's header line, marker normalisation and sort order are not a stable
interface, so CI pins the `uv` version that produced the committed file. A
contributor regenerating with a different local `uv` may produce a diff they
cannot explain; check `uv --version` first. Moving the pin means regenerating
the lock in the same commit.
(`pyproject.toml`'s `maturin` range is a different thing — the build backend
the wheel is built with — and moves on its own schedule.)
