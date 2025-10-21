'''For intercepting and controlling RDKit logging (namely that which is not done in Python)'''
# DEV: this definitely feels like it should be in interfaces.rdkit instead, but doing so would necessitate circular imports
# At some point, ought ot sort out how much chemical functionality we want to expose as RDKit specific or hide the implementation

from typing import Literal, Generator

from rdkit.RDLogger import DisableLog, EnableLog, LogMessage, _levels as RDLoggerNames
from contextlib import contextmanager


@contextmanager
def suppress_rdkit_logs(spec : Literal[*RDLoggerNames]='rdApp.error') -> Generator[None, None, None]:
    '''Temporarily suppress C++ based RDKit log output (useful in conjunction with handling Exceptions thrown by RDKit)'''
    if spec not in RDLoggerNames:
        raise ValueError(f'Logging target must be one of {RDLoggerNames}')
    
    DisableLog(spec)
    yield None # execute "with" block code here
    EnableLog(spec)