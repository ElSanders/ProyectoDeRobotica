from abc import ABC, abstractmethod

class ObjectIdentifier(ABC):
    def __init__(self, img_path):
        self.obs = { 'achieved_goal' : [0,0,0.42470209],
                     'desired_goal' : [0,0,0.42469975]}
        self.img_path_ = img_path
        super().__init__()

    @abstractmethod
    def ProcessImage(self):
        # Process the image located in img_path, identifying the coordinates of
        # the initial position and the final position of the object.
        pass

    @abstractmethod
    def GetCordinates(self):
        # Returns the initial coordinates in the form of a dictionary.
        # ['achieved_goal'] = [x,y,z]
        # ['desired_goal'] = [x,y,z]
        pass
