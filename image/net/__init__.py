from net.squeezenet.squeezenet import Squeezenet


catalogue = dict()


def register(cls):
    catalogue.update({cls.name: cls})

register(Squeezenet)
