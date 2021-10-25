#Required libraries
import numpy as np
import gym
import matplotlib as mpl
mpl.use('Agg')
from numpy.core.shape_base import block
import matplotlib.pyplot as plt
from mujoco_py import GlfwContext
import cv2 
GlfwContext(offscreen=True)
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
#Temporary variables for dummy policy
counter = 0
picked = False
finished = False
ins = []
movement = []

def dummy_policy(obs):
    """Returns array with instructions for robot"""
    global counter
    global picked 
    global finished

    gripper_position = obs['observation'][0:3]
    cube_position = obs['achieved_goal']
    final_position = obs['desired_goal']

    #print(picked)
    if(not picked):
        MSE = ((cube_position[2] - gripper_position[2])**2 + (cube_position[0] - gripper_position[0])**2 + (cube_position[1] - gripper_position[1])**2)/3
        if(MSE <= 0.00015):
            picked = True
    if(not picked):
        #print("grip",gripper_position,"cube",cube_position)
        left = cube_position[1] - gripper_position[1]
        forward = cube_position[0] - gripper_position[0]
        up = cube_position[2] - gripper_position[2]
        grip = 1
    else:
        MSE = ((final_position[2] - gripper_position[2])**2 + (final_position[0] - gripper_position[0])**2 + (final_position[1] - gripper_position[1])**2)/3
        if(MSE > 0.00015):
            #print("grip",gripper_position,"final",final_position)
            left = final_position[1] - gripper_position[1]
            forward = final_position[0] - gripper_position[0]
            up = final_position[2] - gripper_position[2]
            grip = -1
        else:
            finished = True
            return[0,0,0,1]
    #print(forward,left,up,grip)
    return [forward,left,up,grip]


def image_recognition(image):
    """Returns coordinates of gripper, cube, and final position"""
    return [[np.random.random(),np.random.random(),np.random.random()]*3]

def main():
    env = gym.make('FetchPickAndPlace-v1')
    env.seed(12)
    env.reset()
    actions = [0.2,0,0,0]
    for _ in range(500):
        #Obtain image from simulator
        image = env.render(mode='rgb_array')
        #Display image
        cv2.imshow('img',image)
        #16ms delay (60fps)
        cv2.waitKey(16)

        if(not finished):
            #Get observation from our vision function
            #obs = image_recognition(image)
            #Get observation from step function
            obs, reward, done, info = env.step(actions)
            print(f"Rewards: {reward}")
            actions = dummy_policy(obs)
            env.step(actions)
    env.close()
    print("done")

if __name__ == '__main__':
    main()