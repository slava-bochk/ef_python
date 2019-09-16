import pytest

from ef.util.subclasses import get_all_subclasses, Registered


@pytest.fixture
def clear_registered():
    orig = Registered.subclasses
    Registered.subclasses = {}
    yield
    Registered.subclasses = orig


@pytest.mark.usefixtures('clear_registered')
class TestSubclasses:
    def test_global_registered(self):
        class A0(Registered):
            pass

        class B0(Registered):
            pass

        assert A0.subclasses == {'test_subclasses.A0': A0,
                                 'test_subclasses.B0': B0}

    def test_registered_and_subclasses(self):
        class A0(Registered, dont_register=True):
            pass

        class B0():
            pass

        assert A0.subclasses == {}
        assert get_all_subclasses(A0) == set()

        class A1(A0):
            pass

        class B1(B0):
            pass

        class A2(A1):
            pass

        class M1(A0, B1):
            pass

        class M2(A2, B0):
            pass

        assert get_all_subclasses(A0) == {A1, A2, M1, M2}
        assert get_all_subclasses(B0) == {B1, M1, M2}
        assert get_all_subclasses(B1) == {M1}
        assert A0.subclasses == {'test_subclasses.A1': A1,
                                 'test_subclasses.A2': A2,
                                 'test_subclasses.M1': M1,
                                 'test_subclasses.M2': M2}
