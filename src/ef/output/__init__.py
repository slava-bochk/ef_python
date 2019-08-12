from typing import Optional

import h5py
from h5py import File

from ef.simulation import Simulation


class OutputWriter:
    def write(self, sim: Simulation, name: Optional[str] = None) -> None:
        raise NotImplementedError()


class OutputWriterNone(OutputWriter):
    def write(self, sim: Simulation, name: Optional[str] = None) -> None:
        pass


class OutputReader:
    def read(self) -> Simulation:
        raise NotImplementedError()


class OutputWriterNumberedH5(OutputWriter):
    def __init__(self, prefix: str, suffix: str):
        self.prefix: str = prefix
        self.suffix: str = suffix

    def format_filename(self, specific_name: str) -> str:
        return f"{self.prefix}{specific_name}{self.suffix}"

    @classmethod
    def get_filename(cls, sim: Simulation) -> str:
        return f"{sim.time_grid.current_node:07}"

    def write(self, sim: Simulation, name: Optional[str] = None) -> None:
        if name is None:
            name = self.get_filename(sim)
        name = self.format_filename(name)
        print(f"Writing to file {name}")
        with h5py.File(name, 'w') as h5file:
            self.do_write(sim, h5file)

    @classmethod
    def do_write(cls, sim: Simulation, h5file: File) -> None:
        raise NotImplementedError()
