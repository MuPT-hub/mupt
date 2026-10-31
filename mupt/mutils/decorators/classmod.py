"""Decorators for modifying classes"""

from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from typing import Protocol, cast, overload
from functools import wraps

C = TypeVar("C")
type Decorator[**Params, ReturnType] = Callable[
    [Callable[Params, ReturnType]], Callable[Params, ReturnType]
]


def generate_repr(
    cls: Optional[C] = None,
    disp_attrs: Optional[Iterable[str]] = None,
    lookup_attr: Optional[str] = None,
) -> Union[C, Callable[[C], C]]:
    """
    Class decorator for auto-generating __repr__ methods

    By default (i.e. with no arguments), generated repr simply returns name of class
    * If collection of "disp_attrs" is provided, will display the values
      of <disp_attrs> for the object instance being represented in series
    * If "disp_attrs" is NOT provided but "lookup_attr" is,
      disp_attrs will be looked up from the modified parent class
    """
    if disp_attrs is None:
        disp_attrs = []  # set to empty list to avoid mutable default

    def class_decorator(cls: C) -> C:
        """The actual (argument-free) class decorator"""
        # nonlocal avoids multiple scope issues in disparate use cases
        # (refers to the variable in the outermost scope)
        nonlocal disp_attrs

        # only use lookup if one is explicitly provided
        # and no display attributes are provided
        if not disp_attrs and lookup_attr:
            # if a lookup attribute is provided, look up the
            # attribute names within the class being modified
            assert hasattr(cls, lookup_attr)
            disp_attrs = getattr(cls, lookup_attr)

        def _repr_generic(self) -> str:
            attr_str = ", ".join(f"{attr}={getattr(self, attr)}" for attr in disp_attrs)
            return f"{cls.__name__}({attr_str})"

        setattr(cls, "__repr__", _repr_generic)

        return cls

    # null case (i.e. call without parens), return factory call
    if cls is None:
        return class_decorator

    # return literal class decorator call
    return class_decorator(cls)


def register_subclasses(
    cls: Optional[C] = None,
    key_attr: str = "__name__",
    reg_attr: str = "subclass_registry",
) -> Union[C, Callable[[C], C]]:
    """
    Parametric class decorator for automatically
    generating a registry of subclasses of a target class

    Binds registry to the "registry" class property in the target class
    Subclasses are keyed by lookup of a target attribute <key_attr>
    in the child classes (by default just the name of the subclass),
    while the resulting registry class property is bound
    to the <reg_attr> attribute of the parent class
    """

    def class_decorator(cls: C) -> C:
        """The actual (argument-free) class decorator"""

        # property allows dynamic subclassing (generated at runtime, not compile time)
        @classmethod
        @property
        def _registry(cls: C) -> dict[str, C]:
            return {  # Keep a registry of all charger implementations for convenience
                getattr(subclass, key_attr): subclass
                for subclass in cls.__subclasses__()
            }

        # bind registry class property to target class.
        # TODO : check for registry already present in class
        setattr(cls, reg_attr, _registry)

        return cls  # return back the modified class

    if cls is None:  # null case (i.e. call without parens), return factory call
        return class_decorator
    return class_decorator(cls)  # return literal class decorator call


# NOTE: "klass" is needed to distinguish between the class modified by
# this decorator and the classmethod arg when calling super()
# "klass" here is the parent, while "cls" is the child
def register_abstract_class_attrs(
    *attr_names: list[str],
) -> Callable[[C], C]:  # TODO: add mechanism for typehinting
    """
    Register a list of string attribute names as abstract class attributes,
    which MUST be implemented by child classes of the wrapped class
    """

    def class_decorator(klass: C) -> C:
        """The actual (argument-free) class decorator"""

        def __wrapped_init_subclass__(cls: C, **kwargs) -> None:
            """
            Wrapper for subclass definition which actually
            enforces that all named attributes are set
            """
            for attr_name in attr_names:
                # want this removed from kwargs before passing to super,
                # regardless of whether already set in child
                # check if this has been set in the child in code
                passed_attr_value = kwargs.pop(attr_name, NotImplemented)
                attr_val_on_child = getattr(cls, attr_name, NotImplemented)

                # if the value has not been set in code...
                if attr_val_on_child is NotImplemented:
                    # ...fall back to value passed into class definition, if it exists
                    if passed_attr_value is not NotImplemented:
                        setattr(cls, attr_name, passed_attr_value)
                    # ...otherwise, fail and raise Exception
                    else:
                        raise TypeError(
                            f"Can't instantiate abstract class {cls.__name__} with "
                            f"abstract class property '{attr_name}' undefined"
                        )

            super(klass, cls).__init_subclass__(
                **kwargs
            )  # this should fail if extraneous named args are passed

        klass.__init_subclass__ = classmethod(__wrapped_init_subclass__)
        return klass

    # no need for application check here, since the parameterized
    # decorator doesn't take a class to be modified
    return class_decorator


class PreconditionDecorator(Protocol):
    """Type of the generated decorator: bare (`@dec`) or configured (`@dec(msg=...)`)"""

    __name__: str

    @overload
    def __call__[**P, R](self, func: Callable[P, R], /) -> Callable[P, R]: ...
    @overload
    def __call__[**P, R](
        self, /, *, msg: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def precondition_factory(
    flag_attr: str,
    default_msg: str,
    *,
    raise_when: bool = True,
) -> tuple[property, Callable[[Any], None], PreconditionDecorator]:
    """
    Generates a (property, precondition-method, precondition-decorator) triple based on
    some boolean instance attribute which indicates whether other fields are mutable.

    These three callables are ready-made method, which can
    be bound to custom names inside a function to act as:
    * A property which tracks the value of the attribute
    * A private method which raises AttributeError if the attribute equals <raise_when>
    * A decorator which can be applied to other methods in the class body
        to run the precondition before attempting to run the method body

    Boilerplate for mutability check methods based on a "private" attribute.
    """
    flag_attr_public: str = flag_attr.lstrip("_")

    def frozen(self) -> bool:
        is_frozen: bool = getattr(self, flag_attr)
        if not isinstance(is_frozen, bool):
            raise TypeError(
                f"{type(self).__name__}.{flag_attr} must return a bool, "
                f"got {type(is_frozen).__name__} instead"
            )
        return is_frozen

    frozen.__doc__ = f"Whether editing {flag_attr_public} is allowed"

    def _precondition(self, msg: str = default_msg) -> None:
        if frozen(self) is raise_when:
            raise AttributeError(msg)

    _precondition.__name__ = f"_precondition_{flag_attr_public}"
    _precondition.__doc__ = f"Raises AttributeError if {flag_attr} is {raise_when}"

    def precondition_decorator[**P, R](
        func: Callable[P, R] | None = None, /, *, msg: str | None = None
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Configurable decorator which wraps other methods to call
        the precondition herein before running the method body.

        Can be used bare or with a custom message, i.e.:

            @precondition_<flag_attr>
            def do_thing(self): ...

            @precondition_<flag_attr>(msg="Custom error")
            def do_thing(self): ...
        """

        def decorator(f: Callable[P, R], /) -> Callable[P, R]:
            @wraps(f)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                _precondition(args[0], default_msg if msg is None else msg)
                return f(*args, **kwargs)

            return wrapper

        return decorator(func) if callable(func) else decorator

    precondition_decorator.__name__ = f"precondition_{flag_attr_public}"
    precondition_decorator.__doc__ = (
        f"Decorator that guards a method behind a {flag_attr} mutable check"
    )

    return (
        property(frozen),
        _precondition,
        cast(PreconditionDecorator, precondition_decorator),
    )
