import gym
import cv2
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

path = "img/env.png"

# Create a new environment and save an image in path.
def TakePicture(env):
    #  Create a new env Fetch Pick and Place.
    # Save image in path/
    img = env.render(mode="rgb_array")
    img = cv2.resize(img, (560,560), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
    #return [path, observation]