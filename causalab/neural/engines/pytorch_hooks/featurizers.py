"""Moved to :mod:`causalab.neural.shared.featurizers` — applying a featurizer
is engine-neutral tensor math; only a fit's forwards (train.py) are
engine work. This re-export keeps the old import path for one deprecation
beat.
"""

from causalab.neural.shared.featurizers import *  # noqa: F401,F403
