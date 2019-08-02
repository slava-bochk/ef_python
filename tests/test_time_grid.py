from io import BytesIO

import h5py
import pytest
from pytest import raises

from ef.config.components import time_grid
from ef.time_grid import TimeGrid


class TestTimeGrid:
    def test_init(self):
        assert TimeGrid(100.0, 1.0, 10.0) == TimeGrid(100, 1, 10)
        t = TimeGrid(123, 3, 13)
        assert t.total_time == 123
        assert t.total_nodes == 42
        assert t.time_step_size == 3
        assert t.time_save_step == 12
        assert t.node_to_save == 4
        assert t.current_time == 0
        assert t.current_node == 0
        t = TimeGrid(123, 4, 13, 3.14, 1234)
        assert t.total_time == 123
        assert t.total_nodes == 32
        assert t.time_step_size == 123 / 31
        assert t.time_save_step == 123 / 31 * 3
        assert t.node_to_save == 3
        assert t.current_time == 3.14
        assert t.current_node == 1234

    def test_init_exceptions(self):
        with raises(ValueError, match="Expect total_time > 0"):
            TimeGrid(-100, 1, 10)
        with raises(ValueError, match="Expect time_step_size > 0"):
            TimeGrid(100, -1, 10)
        with raises(ValueError, match="Expect time_save_step >= time_step_size"):
            TimeGrid(100, 1, -10)
        with raises(ValueError, match="Expect time_step_size <= total_time"):
            TimeGrid(100, 1000, 1000)
        with raises(ValueError, match="Expect time_save_step >= time_step_size"):
            TimeGrid(100, 10, 1)
        # Not an error, just never save after the 0th tick
        t = TimeGrid(100, 1, 1000)
        assert t.node_to_save == 1000
        assert t.time_save_step == 1000

    def test_dict(self):
        assert TimeGrid(100, 1, 10, 3.14, 123).dict == {'total_time': 100, 'time_step_size': 1, 'time_save_step': 10,
                                                        'current_time': 3.14, 'current_node': 123}

    def test_config_make(self):
        assert time_grid.TimeGridConf().make() == TimeGrid(100.0, 1.0, 10.0)
        assert time_grid.TimeGridConf(123, 13, 3).make() == TimeGrid(123, 3, 13)

    def test_to_component(self):
        assert TimeGrid(100, 1, 10).to_component() == time_grid.TimeGridConf(100, 10, 1)
        assert TimeGrid(123, 3, 13).to_component() == time_grid.TimeGridConf(123, 12, 3)

    def test_update_next_step(self):
        t = TimeGrid(100, 1, 10, 3.14, 123)
        t.update_to_next_step()
        assert t.current_node == 124
        assert t.current_time == pytest.approx(4.14)

    def test_init_h5(self):
        bio = BytesIO()
        grid1 = TimeGrid(123, 3, 13, 3.14, 111)
        with h5py.File(bio, mode="w") as h5file:
            grid1.save_h5(h5file.create_group("/gr"))
        with h5py.File(bio, mode="r") as h5file:
            grid2 = TimeGrid.load_h5(h5file["/gr"])
        assert grid1 == grid2

    def test_import_h5(self, tmpdir):
        bio = BytesIO()
        grid1 = TimeGrid(123, 3, 13, 3.14, 111)
        with h5py.File(bio, mode="w") as h5file:
            grid1.export_h5(h5file.create_group("/gr"))
        with h5py.File(bio, mode="r") as h5file:
            grid2 = TimeGrid.import_h5(h5file["/gr"])
        assert grid1 == grid2

    def test_print(self):
        grid = TimeGrid(100, 1, 10)
        assert str(grid) == ("### TimeGrid:\n"
                             "total_time = 100.0\n"
                             "total_nodes = 101\n"
                             "time_step_size = 1.0\n"
                             "time_save_step = 10.0\n"
                             "node_to_save = 10\n"
                             "current_time = 0.0\n"
                             "current_node = 0")
