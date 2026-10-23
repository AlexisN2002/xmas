"""Model backends: promptolution's APILLM plus ``score`` (teacher-forced log-probabilities) and a per-call log.

:class:`APILLM` can also return call metadata (:class:`CallMeta`) next to its answers.
"""

from xmas.llms.api_llm import APILLM
from xmas.llms.call_log import CallLog, CallRecord
from xmas.llms.meta import CallMeta, SupportsMeta, TokenPrice
from xmas.llms.scoring import ScoredSequence

__all__ = ["APILLM", "CallLog", "CallMeta", "CallRecord", "ScoredSequence", "SupportsMeta", "TokenPrice"]
