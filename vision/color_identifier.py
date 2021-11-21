import cv2
from object_identifier import *
import image_util

class ColorIdentifier(ObjectIdentifier):
    def ProcessImage(self):
        img = cv2.imread(self.img_path_)
        image_util.WorkingArea(img)
        # Identify red ball
        img2 = img.copy()
        red_count = 0.0
        x_red = 0.0
        y_red = 0.0
        for i in range(560):
            for j in range(560):
                if img[i][j][1] == img[i][j][0] and img[i][j][0] == img[i][j][2]:
                    img[i][j] = [255,255,255]
                else:
                    red_count += 1
                    x_red += i
                    y_red += j
        self.obs['desired_goal'][0], self.obs['desired_goal'][1] = image_util.ToRealWorld(x_red / red_count, y_red / red_count)
        # Identify black square
        img = img2
        black_count = 0.0
        x_black = 0.0
        y_black = 0.0
        for i in range(560):
            for j in range(560):
                if img[i][j][1] != img[i][j][0] or img[i][j][2] > 130:
                    img[i][j] = [255,255,255]
                else:
                    black_count += 1
                    x_black += i
                    y_black += j
        self.obs['achieved_goal'][0], self.obs['achieved_goal'][1] = image_util.ToRealWorld(x_black / black_count, y_black / black_count)



    def GetCordinates(self, obs):
        if (self.cont == 0):
            self.cont += 1
            return self.obs
        return obs