"""Unit tests for decorators which modify classes"""

import pytest

from typing import Any
import re

from mupt.mutils.decorators.classmod import frozen_precondition_factory


# Example classes
DEFAULT_MSG: str = "Hierarchy is frozen"
CUSTOM_MSG: str = "Custom error"
SUB_MSG: str = "Sub custom"


class Base:
    """Example class with preconditions created using frozen_precondition_factory"""

    _frozen_hierarchy: bool

    def __init__(self, frozen: bool = False) -> None:
        self._frozen_hierarchy = frozen

    (
        frozen_hierarchy,
        _precondition_mutable_hierarchy,
        precondition_mutable_hierarchy,
    ) = frozen_precondition_factory("_frozen_hierarchy", DEFAULT_MSG, raise_when=True)

    @precondition_mutable_hierarchy
    def guarded_bare(self) -> str:  # noqa: D102
        return "ok"

    @precondition_mutable_hierarchy(msg=CUSTOM_MSG)
    def guarded_with_msg(self) -> str:  # noqa: D102
        return "ok"


class Sub(Base):
    """Example subclass to test that preconditions survive inheritance"""

    @Base.precondition_mutable_hierarchy
    def sub_guarded(self) -> str:  # noqa: D102
        return "sub_ok"

    @Base.precondition_mutable_hierarchy(msg=SUB_MSG)
    def sub_guarded_with_msg(self) -> str:  # noqa: D102
        return "sub_ok"


class Multi:
    """Example class with multiple precondtiions to test independence"""

    _a: bool = False
    _b: bool = False
    frozen_a, _precondition_a, precondition_a = frozen_precondition_factory(
        "_a", "A frozen.", raise_when=True
    )
    frozen_b, _precondition_b, precondition_b = frozen_precondition_factory(
        "_b", "B frozen.", raise_when=True
    )

    def __init__(self, a: bool, b: bool) -> None:
        self._a = a
        self._b = b

    @precondition_a
    def op_a(self) -> str:  # noqa: D102
        return "a"

    @precondition_b
    def op_b(self) -> str:  # noqa: D102
        return "b"


# Helpers
def exactly(msg: str) -> str:
    """Regex for pytest.raises(match=...) that matches <msg> and nothing else."""
    return f"^{re.escape(msg)}$"


# Tests proper
def test_factory_metadata() -> None:
    """
    Test that dynamically-created method names and docstring
    have the target attribute interpolated into the correctly
    """
    prop, pre, dec = frozen_precondition_factory("___flag", "msg")
    assert isinstance(prop, property)
    assert prop.__doc__ == "Whether editing flag is allowed"
    assert pre.__name__ == "_precondition_flag"
    assert dec.__name__ == "precondition_flag"


def test_decorator_ignores_precondition_override() -> None:
    """
    Test that, under the hood, the decorator calls the precondition of the class
    on which it was defined, even if there is a local override in a subclass
    """

    class C(Base):
        def _precondition_mutable_hierarchy(self, msg: str = DEFAULT_MSG) -> None:
            raise RuntimeError(f"overridden: {msg}")

        @Base.precondition_mutable_hierarchy
        def do_thing(self) -> str:
            return "ok"

    with pytest.raises(AttributeError):
        C(frozen=True).do_thing()


class TestProperty:
    """Bundled tests for the property created over the target attribute"""

    @pytest.mark.parametrize("frozen", [False, True])
    def test_property_tracks_attribute(self, frozen: bool) -> None:
        """
        Test that the created property reports the
        current value of the tracked attribute
        """
        obj = Base(frozen)
        assert obj.frozen_hierarchy is frozen

        obj._frozen_hierarchy = not frozen
        assert obj.frozen_hierarchy is (not frozen)

    @pytest.mark.parametrize(
        "bad_value, match",
        [(1, "int"), (None, "NoneType"), ("yes", "str")],
    )
    def test_property_rejects_invalid_type(self, bad_value: Any, match: str) -> None:
        """Test that invalid types for the property value are caught and rejected"""
        b_inst = Base()
        b_inst._frozen_hierarchy = bad_value

        with pytest.raises(TypeError, match=match):
            _ = b_inst.frozen_hierarchy

    def test_precondition_rejects_no_attribute(self) -> None:
        """Test that the property errs if the target attribute is never defined"""
        b_inst = Base()
        del b_inst._frozen_hierarchy

        with pytest.raises(AttributeError, match="object has no attribute"):
            _ = b_inst.frozen_hierarchy


class TestPrecondition:
    """Bundled test for checking the preconditions are enforced as configured"""

    def test_precondition(self) -> None:
        """
        Test that the 'private' precondition raises Exception on
        the target of raise_when and is silent otherwise
        """
        Base(frozen=False)._precondition_mutable_hierarchy()
        with pytest.raises(AttributeError, match=exactly(DEFAULT_MSG)):
            Base(frozen=True)._precondition_mutable_hierarchy()

    @pytest.mark.parametrize("raise_when", [False, True])
    def test_precondition_raise_when(self, raise_when: bool) -> None:
        """Test that the condition on which to raise Exception is adhered to"""

        class Dummy:
            def __init__(self, thing: bool) -> None:
                self._thing = thing

            thing_set, _precondition_thing, precondition_thing = (
                frozen_precondition_factory("_thing", "msg", raise_when=raise_when)
            )

            @precondition_thing
            def greet(self) -> str:
                return "hello"

        d = Dummy(not raise_when)
        _ = d.greet()

        with pytest.raises(AttributeError, match=exactly("msg")):
            d = Dummy(raise_when)
            _ = d.greet()


class TestDecorator:
    """Bundled tests for the convenient decorator the factory produces"""

    @pytest.mark.parametrize(
        "name, result, msg",
        [
            ("guarded_bare", "ok", DEFAULT_MSG),
            ("guarded_with_msg", "ok", CUSTOM_MSG),
            ("sub_guarded", "sub_ok", DEFAULT_MSG),
            ("sub_guarded_with_msg", "sub_ok", SUB_MSG),
        ],
    )  # test on Sub to also cover Base as a second bird for this one stone
    def test_decorated_method(self, name: str, result: str, msg: str) -> None:
        """
        Test that decorator created on example classes fits in with its
        containing class and produces the expected value/Exception on known inputs
        """
        assert getattr(Sub, name).__name__ == name  # check result of wraps() call

        assert getattr(Sub(frozen=False), name)() == result
        with pytest.raises(AttributeError, match=exactly(msg)):
            getattr(Sub(frozen=True), name)()

    def test_decorator_runs_before_method_body(self) -> None:
        """Test that the decorated method is not run if the precondition is unmet"""
        calls: list[int] = []

        class C(Base):
            @Base.precondition_mutable_hierarchy
            def add(self, x: int, *, y: int = 0) -> int:
                calls.append(x)
                return x + y

        assert C().add(2, y=3) == 5
        with pytest.raises(AttributeError):
            C(frozen=True).add(1)
        assert calls == [2]

    @pytest.mark.parametrize(
        "a,b", [(False, False), (False, True), (True, False), (True, True)]
    )
    def test_factories_are_independent(self, a: bool, b: bool) -> None:
        """Test that multiple wrapped preconditions do not affect one another"""
        obj = Multi(a, b)
        for op, frozen, msg, result in [
            (obj.op_a, a, "A frozen.", "a"),
            (obj.op_b, b, "B frozen.", "b"),
        ]:
            if frozen:
                with pytest.raises(AttributeError, match=exactly(msg)):
                    op()
            else:
                assert op() == result
