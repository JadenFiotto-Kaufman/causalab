"""The gated delta rule for a sequence that fits one chunk — the shape the
training and evaluation forwards of Qwen3.6-A3B always take (13 tokens).

FLA's chunked kernels pad every sequence to a 64-token chunk and run the
whole inter-chunk machinery (the state carried between chunks and its
gradient) for a single chunk started from a zero state, where that half is
identically zero work. This package holds the single-chunk closed form: a
float32 torch reference (:mod:`.reference`), a Triton kernel with a full
backward (:mod:`.triton_kernel`), the per-forward binding that routes
short sequences to it while longer ones keep FLA (:mod:`.binding`), and the
threshold that binding reads from the environment (:mod:`.options`).
"""

from causalab.neural.shared.gdn_short.binding import (
    selects_single_chunk,
    short_seq_kernel_path,
)
from causalab.neural.shared.gdn_short.options import ShortSeqKernelOptions
from causalab.neural.shared.gdn_short.reference import (
    recurrent_gated_delta_rule_reference,
    single_chunk_gated_delta_rule_torch,
)
from causalab.neural.shared.gdn_short.triton_kernel import (
    MAX_SEQ_LEN,
    single_chunk_gated_delta_rule,
    triton_available,
)

__all__ = [
    "MAX_SEQ_LEN",
    "ShortSeqKernelOptions",
    "recurrent_gated_delta_rule_reference",
    "selects_single_chunk",
    "short_seq_kernel_path",
    "single_chunk_gated_delta_rule",
    "single_chunk_gated_delta_rule_torch",
    "triton_available",
]
