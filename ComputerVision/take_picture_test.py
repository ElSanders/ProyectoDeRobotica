import gym
import cv2

path = "ComputerVision/img/test5.png"

# Create a new environment and save an image in path.
def TakePicture():
    #  Create a new env Fetch Pick and Place.
    env = gym.make('FetchPickAndPlace-v1')
    observation = env.reset()
    # Save image in path/
    img = env.render(mode="rgb_array")
    img = cv2.resize(img, (560,560), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
    print("Hello ")
    env.close()
    return [path, observation] 


p, obs = TakePicture()
f = open("ComputerVision/obs/test5.txt", "w")
f.write(str(obs))
f.close()