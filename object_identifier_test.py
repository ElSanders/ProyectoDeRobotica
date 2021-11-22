from ComputerVision.object_identifier_factory import ObjectIdentifierFactory
import time


def GetError(real, measure):
    return abs((real - measure) / real) * 100


def UnitTest():
    print("\t\t\tVersion 1 (color)\t\t\tVersion 2 (shape)")
    print("# Test\t\tTiempo\t\t\tError\t\tTiempo\t\t\tError")
    color_time_avg = 0.0
    color_error_avg = 0.0
    shape_time_avg = 0.0
    shape_error_avg = 0.0
    for i in range(1, 6):
        # Leer file
        text_path = "ComputerVision/obs/test" + str(i) + ".txt"
        f = open(text_path, "r")
        str_obs = f.readlines()
        achieved_goal = list(map(float, str_obs[0].split()))
        desired_goal = list(map(float, str_obs[1].split()))
        f.close()
        # Usar color_identifier
        img_path = "ComputerVision/img/test" + str(i) + ".png"
        color_identifier = ObjectIdentifierFactory("color", img_path)
        color_s_time = time.time()
        color_identifier.ProcessImage()
        color_e_time = time.time()
        color_cordinates = color_identifier.GetCordinates("")
        color_error_Sum = 0.0
        for j in range(3):
            color_error_Sum += GetError(achieved_goal[j],
                                        color_cordinates['achieved_goal'][j])
        for j in range(3):
            color_error_Sum += GetError(desired_goal[j],
                                        color_cordinates['desired_goal'][j])
        color_error_Sum /= 6
        color_time_avg += color_e_time - color_s_time
        color_error_avg += color_error_Sum
        # Usar shape identifier
        shape_identifier = ObjectIdentifierFactory("shape", img_path)
        shape_s_time = time.time()
        shape_identifier.ProcessImage()
        shape_e_time = time.time()
        shape_cordinates = shape_identifier.GetCordinates("")
        shape_error_sum = 0.0
        for j in range(3):
            shape_error_sum += GetError(achieved_goal[j],
                                        shape_cordinates['achieved_goal'][j])
        for j in range(3):
            shape_error_sum += GetError(desired_goal[j],
                                        shape_cordinates['desired_goal'][j])
        shape_error_sum /= 6
        shape_time_avg += shape_e_time - shape_s_time
        shape_error_avg += shape_error_sum
        print(i, "\t\t", "{:.2f}".format(color_e_time - color_s_time),
              "s\t\t\t", "{:.2f}".format(color_error_Sum), "%\t\t", "{:.2f}".format(
                  shape_e_time - shape_s_time),
              "s\t\t\t", "{:.2f}".format(shape_error_sum), "%")
    color_time_avg /= 5
    color_error_avg /= 5
    shape_time_avg /= 5
    shape_error_avg /= 5
    print("avg\t\t", "{:.2f}".format(color_time_avg), "s\t\t\t", "{:.3f}".format(color_error_avg),
          "%\t", "{:.2f}".format(shape_time_avg), "s\t\t\t", "{:.3f}".format(shape_error_avg), "%")
    print("Mejora\t\t\t\t\t\t\t", "{:.2f}".format((color_time_avg - shape_time_avg) / color_time_avg * 100.00), "%")


if __name__ == '__main__':
    UnitTest()
