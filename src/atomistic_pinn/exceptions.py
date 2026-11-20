"""Exception hierarchy for Atomistic PINN Lab.

All package-specific exceptions derive from :class:`AtomisticPINNError` so
callers can catch the whole family with a single ``except`` clause. The
hierarchy is intentionally small; new exceptions are added when a sprint
actually needs them, not speculatively.
"""

from __future__ import annotations


class AtomisticPINNError(Exception):
    """Base class for all errors raised by this package."""


class DataContractError(AtomisticPINNError):
    """Raised when input data violates a documented data contract.

    Covers, for example, datasets whose structure identifiers are not unique
    or files whose contents do not match the schema in
    ``atomistic_pinn.data.schema``. Pydantic ``ValidationError`` is raised for
    schema-level issues; this exception is used for contract-level checks
    beyond single-structure validation.
    """


class ConfigurationError(AtomisticPINNError):
    """Raised when an experiment configuration is invalid or incomplete.

    In particular, this covers configurations that silently fill in values
    the source paper does not specify (see ``configs/*.yaml`` and
    ``docs/equations.md`` § "Known uncertainties").
    """
