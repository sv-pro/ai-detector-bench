"""detbench — a referee for AI-text detectors.

The claim this package makes is not "we detect AI-generated text". It is: *we measure
detectors, including our own, at the operating points that matter, and we publish where
they fail.*

Everything here follows from that. Refusal is a first-class result. Scores are not
probabilities until calibrated. The headline metric is true-positive rate at a defensible
false-positive rate, not AUROC. And every detector is scored twice — once on clean text,
once under attack — because the gap between those two numbers is the finding.
"""

from .core import Detector, RefusalReason, Verdict, token_count
from .metrics import Report, auroc, evaluate, overconfidence_rate, tpr_at_fpr

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "Verdict",
    "RefusalReason",
    "token_count",
    "Report",
    "evaluate",
    "auroc",
    "tpr_at_fpr",
    "overconfidence_rate",
    "__version__",
]
