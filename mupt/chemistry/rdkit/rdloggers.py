"""
For intercepting and controlling RDKit logging,
namely that which is not done in Python
"""

from typing import Literal, Generator

from rdkit.RDLogger import DisableLog, EnableLog, _levels as RDLoggerNames
from contextlib import contextmanager


@contextmanager
def suppress_rdkit_logs(
    spec: Literal[*RDLoggerNames] = "rdApp.error",
) -> Generator[None, None, None]:
    """
    Temporarily suppress C++ based RDKit log output
    Useful in conjunction with handling Exceptions thrown by RDKit
    """
    if spec not in RDLoggerNames:
        raise ValueError(f"Logging target must be one of {RDLoggerNames}")

    DisableLog(spec)
    yield None  # execute "with" block code here
    EnableLog(spec)
