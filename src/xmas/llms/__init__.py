"""Model backends.

Each backend subclasses its promptolution counterpart and adds ``score``, the
teacher-forced sibling of generation. :class:`APILLM` can also return call
metadata (:class:`CallMeta`) next to its answers.
"""

from xmas.llms.api_llm import APILLM, RateLimitedError, ScoringUnsupportedError
from xmas.llms.masking import assistant_span, mask_logprobs
from xmas.llms.meta import CallMeta, SupportsMeta, TokenPrice
from xmas.llms.scoring import ScoredSequence, SupportsScore
from xmas.llms.telemetry import CallLog, CallRecord, Price

__all__ = [
    "APILLM",
    "CallLog",
    "CallMeta",
    "CallRecord",
    "Price",
    "RateLimitedError",
    "ScoredSequence",
    "ScoringUnsupportedError",
    "SupportsMeta",
    "SupportsScore",
    "TokenPrice",
    "assistant_span",
    "mask_logprobs",
]
