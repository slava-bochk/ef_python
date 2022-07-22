import ast
import operator as op

import inject
import numpy
from simpleeval import SimpleEval

from ef.field import Field


class FieldExpression(Field):

    @inject.params(xp=numpy)
    def __init__(self, name, electric_or_magnetic, expression_x, expression_y, expression_z, xp=numpy):
        super().__init__(name, electric_or_magnetic)
        self._xp = xp
        self.expression_x = expression_x
        self.expression_y = expression_y
        self.expression_z = expression_z
        self._ev = SimpleEval(functions={"sin": xp.sin,
                                         "cos": xp.cos,
                                         "sqrt": xp.sqrt},
                              operators={ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                                         ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
                                         ast.Pow: op.pow, ast.Mod: op.mod,
                                         ast.Eq: op.eq, ast.NotEq: op.ne,
                                         ast.Gt: op.gt, ast.Lt: op.lt,
                                         ast.GtE: op.ge, ast.LtE: op.le,
                                         ast.Not: op.not_,
                                         ast.USub: op.neg, ast.UAdd: op.pos,
                                         ast.In: lambda x, y: op.contains(y, x),
                                         ast.NotIn: lambda x, y: not op.contains(y, x),
                                         ast.Is: lambda x, y: x is y,
                                         ast.IsNot: lambda x, y: x is not y,
                                         }
                              )
        # todo: inherit SimpleEval and define math functions inside
        # todo: add r, theta, phi names

    def get_at_points(self, positions, time: float) -> numpy.ndarray:
        positions = self._xp.asarray(positions)
        self._ev.names["t"] = time
        self._ev.names["x"] = positions[:, 0]
        self._ev.names["y"] = positions[:, 1]
        self._ev.names["z"] = positions[:, 2]
        result = self._xp.empty_like(positions)
        result[:, 0] = self._ev.eval(self.expression_x)
        result[:, 1] = self._ev.eval(self.expression_y)
        result[:, 2] = self._ev.eval(self.expression_z)
        return result
