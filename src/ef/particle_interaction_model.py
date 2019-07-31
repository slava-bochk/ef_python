from enum import Enum, auto


class Model(Enum):
    noninteracting = auto()
    binary = auto()
    PIC = auto()

    def __repr__(self):
        return f"Model.{self.name}"
