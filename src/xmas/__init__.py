"""xmas: a lightweight framework for building and optimizing multi-agent systems.

xmas is promptolution's abstractions lifted one level. The multi-agent system is
the predictor, and its per-agent prompts, its topology and its per-node model
bindings are components an optimizer can improve. A one-agent system is not an
analogy to a promptolution predictor; it is one.

Note:
    The build reads ``__version__`` from this file (see ``[tool.hatch.version]``
    in ``pyproject.toml``), so keep the assignment a plain string literal.
"""

from xmas.agent import Agent
from xmas.budget import Budget
from xmas.edges import Edge
from xmas.errors import BadFormat, BudgetExceeded, ErrorKind, RunStatus
from xmas.graph import END, START
from xmas.health import Health, HealthSettings
from xmas.llms import APILLM, ScoredSequence
from xmas.mas import MAS
from xmas.nodes import Step
from xmas.tasks import GSM8KTask, MASTask, ProbingTask, ReferencePool
from xmas.trace import AgentRecord, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType

__version__ = "0.0.0"

__all__ = [
    "APILLM",
    "END",
    "JSON",
    "MAS",
    "START",
    "TEXT",
    "Agent",
    "AgentRecord",
    "BadFormat",
    "Budget",
    "BudgetExceeded",
    "Edge",
    "ErrorKind",
    "GSM8KTask",
    "Health",
    "HealthSettings",
    "MASTask",
    "ProbingTask",
    "ReferencePool",
    "RunStatus",
    "ScoredSequence",
    "Step",
    "TokenUsage",
    "ToolCall",
    "Trace",
    "ValueType",
    "__version__",
]
