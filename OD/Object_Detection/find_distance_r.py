import math


def circle(r):
    result = []
    for i in range(0, 50):
        theta = i * math.pi / 10-math.pi/2
        dy, dx = round(r * math.sin(theta)), round(r * math.cos(theta))
        repeat = 0
        for points in result:
            if points[0] == dy and points[1] == dx:
                repeat = 1
        if repeat == 0:
            result.append([dy, dx])
    return result


r = 4
print(circle(r))
print(len(circle(r)))
