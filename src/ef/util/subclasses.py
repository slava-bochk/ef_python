from typing import Type, Dict


def get_all_subclasses(cls):
    subclasses = set()
    todo = [cls]
    while todo:
        parent = todo.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                todo.append(child)
    return subclasses


class Registered:
    subclasses: Dict[str, Type['Registered']] = {}

    @classmethod
    def class_key(cls):
        return f"{cls.__module__}.{cls.__name__}"

    def __init_subclass__(cls, dont_register=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not dont_register:
            cls.subclasses[cls.class_key()] = cls
