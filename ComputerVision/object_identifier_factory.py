from ComputerVision.object_identifier import *
from ComputerVision.color_identifier import *
from ComputerVision.shape_identifier import *

# Given the type of identifier and the path, return a new instance of it.
def ObjectIdentifierFactory(type_identifier, img_path):
    if type_identifier == "color":
        return ColorIdentifier(img_path)
    elif type_identifier == "shape":
        return ShapeIdentifier(img_path)
    return None