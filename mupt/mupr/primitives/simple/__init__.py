"""
Concrete Primitive subclass for 'simples', which represent
the bottommost, irreducible units of a molecular system

Used as this direct holders of Connector - provide source of truth for system
"""

from .simple import SimplePrimitive as SimplePrimitive
from .atomic import AtomicPrimitive as AtomicPrimitive
