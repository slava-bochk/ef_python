__all__ = ["ParticleInteractionModelConf", "ParticleInteractionModelSection"]

from collections import namedtuple

from ef.config.component import ConfigComponent
from ef.config.section import ConfigSection
from ef.particle_interaction_model import Model


class ParticleInteractionModelConf(ConfigComponent):
    def __init__(self, model="PIC"):
        if model not in ("PIC", 'noninteracting', 'binary'):
            raise ValueError("Unexpected particle interaction model: {}".format(model))
        self.model = model

    def to_conf(self):
        return ParticleInteractionModelSection(self.model)

    def make(self):
        return Model[self.model]


class ParticleInteractionModelSection(ConfigSection):
    section = "ParticleInteractionModel"
    ContentTuple = namedtuple("ParticleInteractionModelTuple", ('particle_interaction_model',))
    convert = ContentTuple(str)

    def make(self):
        return ParticleInteractionModelConf(self.content.particle_interaction_model)
