from struct_tools import *

if __name__ == "__main__":
    # Note: this setup is purposely messy,
    # in order to test recursion treatments.
    import numpy as np

    a = np.arange(24)
    A = a.reshape((4, -1))

    d1 = dict(a=a, b=99, z=3, A=A)
    d2 = dict(x=1, lorem="ipsum")

    # Dont move this block below (coz then it will contain
    # d1/d2 recursions, rather than a1/a2)
    a1 = AlignedDict(d1)
    a2 = AlignedDict(d2)

    # Json -- cannot handle recursions
    # import json

    d1["d2"] = d2
    # print("\njson.dumps:\n================")
    # print(json.dumps(d1, indent=4, default=repr))

    # pprint
    # import pprint

    d1["d2"] = d2
    d2["d1"] = d1
    d1["lst"] = [0, 1, d2]
    # print("\npprint:\n================")
    # pprint.pprint(d1,compact=True)

    # Regular dict/print
    # print("\nRegular dict/print:\n================")
    # print(d1)

    # Add recursions similar to d1/d2
    # print("\nAlignedDict:\n================")
    a2["a1"] = a1
    a1["a2"] = a2
    a1["one"] = AlignedDict(item="hello")
    a1["empty"] = AlignedDict()
    a1["really long name that goes on and on"] = [0, 1, a2]

    a1.printopts = dict(
        excluded={"z"},
        aliases={"A": "aMatrix"},
        ordering="line",  # or alpha or ["a2", "self"]
        # reverse=True,
    )
    print("\nstr:\n================")
    print(a1)
    print("\nrepr:\n================")
    print(repr(a1))

    print("\n================\nwith const. indent:\n================")

    a1.printopts["indent"] = 1
    a2.printopts = dict(indent=1)
    print("\nstr:\n================")
    print(a1)
    print("\nrepr:\n================")
    print(repr(a1))

    print("\nNicePrint:\n================")

    class MyClass(NicePrint):
        printopts = NicePrint.printopts.copy()

        def __init__(self):
            self._a = 99
            self.a = np.arange(24)
            # self.a = 1
            # self.lorem = "ipsum"
            # self.lst = np.arange(20)

    obj1 = MyClass()
    # obj1.obj2 = MyClass()
    # obj1.self = obj1
    obj1.printopts["excluded"] |= {"lst"}
    print(repr(obj1))
    print(obj1)

    print("\nDotDict:\n================")
    dd = DotDict(a=a, b=99, z=3, A=A)
    dd.dd2 = DotDict(a=a, b=99, z=3, A=A)
    dd.self = dd
    dd.printopts["excluded"] |= {"A"}
    print(repr(dd))
    print(dd)

    # Other tests
    print("\ndeep_getattr:\n================")
    key2 = "self.self.self.self.a"
    print(key2, ":", deep_getattr(dd, key2))
