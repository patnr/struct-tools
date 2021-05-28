from struct_tools import __version__

from struct_tools import DotDict
from copy import deepcopy


def test_copy():
    dct = DotDict(
        inner = DotDict(a="val of a", b="val of b")
    )

    # Create infinite recursion
    dct.inner["dct"] = dct

    # Copy
    new = deepcopy(dct)
    # print(repr(new))

    # Check that the infinite cycle (of new) is contained within new
    assert new.inner.dct is new
    assert new.inner.dct is not dct

    # Make sure this does not affect old values
    new.inner.a = "new a value"
    assert dct.inner.a == "val of a"
