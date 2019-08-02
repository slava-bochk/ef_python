import logging
from math import floor, ceil

from ef.config.components import time_grid
from ef.util.serializable_h5 import SerializableH5


class TimeGrid(SerializableH5):
    def __init__(self, total_time, time_step_size, time_save_step, current_node=0):
        if total_time <= 0:
            raise ValueError("Expect total_time > 0")
        if time_step_size <= 0:
            raise ValueError("Expect time_step_size > 0")
        if time_step_size > total_time:
            raise ValueError("Expect time_step_size <= total_time")
        if time_save_step < time_step_size:
            raise ValueError("Expect time_save_step >= time_step_size")
        self.total_time = float(total_time)
        self.total_nodes = ceil(self.total_time / time_step_size) + 1
        if self.time_step_size != time_step_size:
            logging.warning("Reducing time step to {:.3E} from {:.3E} "
                            "to fit a whole number of cells."
                            .format(self.time_step_size, time_step_size))
        self.node_to_save = floor(time_save_step / self.time_step_size)
        if self.time_save_step != time_save_step:
            logging.warning("Reducing save time step to {:.3E} from {:.3E} "
                            "to fit a whole number of cells."
                            .format(time_save_step, time_save_step))
        self.current_node = current_node

    @property
    def dict(self):
        d = super().dict
        d['time_step_size'] = self.time_step_size
        d['time_save_step'] = self.time_save_step
        del d['total_nodes'], d['node_to_save']
        return d

    @property
    def time_step_size(self):
        return self.total_time / (self.total_nodes - 1)

    @property
    def time_save_step(self):
        return self.node_to_save * self.time_step_size

    @property
    def current_time(self):
        return self.current_node * self.time_step_size

    def to_component(self):
        return time_grid.TimeGridConf(self.total_time, self.time_save_step, self.time_step_size)

    def update_to_next_step(self):
        self.current_node += 1

    @property
    def should_save(self):
        return self.current_node % self.node_to_save == 0

    @staticmethod
    def import_h5(g):
        ga = g.attrs
        return TimeGrid(float(ga['total_time']), float(ga['time_step_size']), float(ga['time_save_step']),
                        int(ga['current_node']))

    def export_h5(self, g):
        for k in ['total_time', 'current_time', 'time_step_size', 'time_save_step', 'total_nodes', 'current_node',
                  'node_to_save']:
            g.attrs[k] = [getattr(self, k)]
