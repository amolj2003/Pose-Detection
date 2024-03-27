import math
# import json

# f = open("assets/exercise.json", "r")  # Open the file in read
# LOAD = json.load(f)
# curls = dict(LOAD["curls"])
# print(curls)

def getAngles(a,b,c):
    # a = landmarks[0] 
    # b = landmarks[1]
    # c = landmarks[2]

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(radians)
    angle = angle + 360 if angle < 0 else angle
    return angle
