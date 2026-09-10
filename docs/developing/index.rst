Developing
==========

Documenting the public API
--------------------------

The Reference section is generated from the Python source during every Sphinx
build. Write NumPy-style docstrings alongside public classes, functions,
methods, and attributes; no separate API page or inventory entry is needed.

Non-underscore members in the supported ``builders``, ``chemistry``,
``geometry``, ``interfaces``, ``mupr``, and ``roles`` namespaces are included
automatically. Keep implementation details in underscore-prefixed members or
private modules. Adding another supported top-level package requires updating
``public_api_prefixes`` in ``docs/conf.py``.
