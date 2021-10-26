import unittest
from shape_identifier import *
from object_identifier_factory import *
from take_picture import *

def GetError(real, measure):
    return abs((real - measure) / real) * 100

# 1 percent error is accepted.
Error = 1.0

class ShapeIdentifier(unittest.TestCase):
    def test_cordinates(self):
        path, obs = TakePicture()
        object_identifier = ObjectIdentifierFactory("shape", path)
        object_identifier.ProcessImage()
        cordinates = object_identifier.GetCordinates()
        for i in range(3):
            self.assertLess(GetError(obs['achieved_goal'][i], cordinates['achieved_goal'][i]), Error)
        for i in range(3):
            self.assertLess(GetError(obs['desired_goal'][i], cordinates['desired_goal'][i]), Error)

if __name__ == '__main__':
    unittest.main()