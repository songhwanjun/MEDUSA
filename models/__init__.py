""" Build a MEDUSA model """

from models.medusa.detr import build as medusa_build

def build_model(args):

    return medusa_build(args)
