from random import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'


def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr


def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr


def goMatr(ang, ang_2):
    mtr = np.array([[1, 0, ang], [0, 1, ang_2], [0, 0, 1]])
    return mtr


def scalingMatr(i):
    mtr = np.array([[i, 0, 0], [0, i, 0], [0, 0, 1]])
    return mtr


def to_proj_coords(x):
    r, c = x.shape
    x = np.concatenate([x, np.ones((1, c))], axis=0)
    return x


def to_cart_coords(x):
    x = x[:-1] / x[-1]
    return x


def Center(x1, x2, x3, y1, y2, y3):
    x = round((x1 + x2 + x3) / 3, 2)
    y = round((y1 + y2 + y3) / 3, 2)

    return x, y


# рисуем пиксель 1
def set_pixel_first(_img, x, y, color):
    _img[256 - x - 1][256 - y - 1] = color


# рисуем пиксель 2
def set_pixel_second(_img, x, y, color):
    _img[256 - y - 1][256 - x - 1] = color


# Алгоритм Брезенхема
def line(_img, x0, y0, x1, y1, color):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dy = abs(y1 - y0)
    signy = -1 if y0 > y1 else 1
    error = 0
    y = y0
    for x in range(x0, x1):
        if steep:
            if x > 0 and y > 0:
                set_pixel_first(_img, x, y, color)
        else:
            if x > 0 and y > 0:
                set_pixel_second(_img, x, y, color)
        error += dy
        if abs(x1 - x0) <= 2 * error:
            y += signy
            error -= abs(x1 - x0)


def draw_circle(_img, x, y, r, color):
    disp_x = x
    disp_y = y
    x = 0
    y = r
    delta = (1 - 2 * r)
    error = 0
    while y >= 0:
        set_pixel_first(_img, disp_x + x, disp_y + y, color)
        set_pixel_first(_img, disp_x + x, disp_y - y, color)
        set_pixel_first(_img, disp_x - x, disp_y + y, color)
        set_pixel_first(_img, disp_x - x, disp_y - y, color)

        error = 2 * (delta + y) - 1
        if (delta < 0) and (error <= 0):
            x += 1
            delta = delta + (2 * x + 1)
            continue
        error = 2 * (delta - x) - 1
        if (delta > 0) and (error > 0):
            y -= 1
            delta = delta + (1 - 2 * y)
            continue
        x += 1
        delta = delta + (2 * (x - y))
        y -= 1


pt0 = [5, 25]
pt1 = [5, 5]  # Вершины треугольника
pt2 = [20, 15]

sq1 = [20, 20]
sq2 = [20, 236]  # Вершины Квадрата
sq3 = [236, 20]
sq4 = [236, 236]

R = 40  # Радиус нашей окружности

x = np.array([pt0, pt1, pt2], dtype=np.float32).T  # Картинка
x_proj = to_proj_coords(x)
x_new = x

collision_count = 0

center = Center(pt0[0], pt1[0], pt2[0], pt0[1], pt1[1], pt2[1])  # Центр треугольника
T_go_center = shiftMatr((150, 180))
N = 300  # Frames
scaling_num = 1
size = 256
color_triagle = np.array([255, 0, 0], dtype=np.uint8)
color_circle = np.array([0, 0, 255], dtype=np.uint8)
color_square = np.array([0, 255, 0], dtype=np.uint8)

# drawing
frames = []
fig = plt.figure()

Qq = False
Qq_2 = False
dx = 0
dy = 0

# Направление и ускорение движения треугольника
go_1 = 1
go_2 = 2


def colculate_matrix(i_, i_1, T_go_center_, Scale_, Rotation_, T_, x_proj_):
    Go_ = goMatr(i_, i_1)
    x_proj_new_ = T_go_center_ @ Go_ @ Scale_ @ np.linalg.inv(T_) @ Rotation_ @ T_ @ x_proj_

    return x_proj_new_


def update_color_for_circle():
    global color_circle
    global color_triagle

    color_buff = color_triagle
    color_triagle = color_circle
    color_circle = color_buff


def update_color_for_square():
    global color_square
    global color_triagle

    color_buff = color_triagle
    color_triagle = color_square
    color_square = color_buff


def circle_touch(x_new_):
    global go_1
    global go_2
    global Qq_2
    global color_circle
    global color_triagle

    Qq_2 = False

    if (x_new_[0, 0] - int(size / 2)) ** 2 + (x_new_[1, 0] - int(size / 2)) ** 2 <= (R + 4) ** 2:
        Qq_2 = True
        go_1 *= -1
        go_2 *= -1
        update_color_for_circle()

    if (x_new_[0, 1] - int(size / 2)) ** 2 + (x_new_[1, 1] - int(size / 2)) ** 2 <= (R + 4) ** 2:
        Qq_2 = True
        go_1 *= -1
        go_2 *= -1
        update_color_for_circle()

    if (x_new_[0, 2] - int(size / 2)) ** 2 + (x_new_[1, 2] - int(size / 2)) ** 2 <= (R + 4) ** 2:
        Qq_2 = True
        go_1 *= -1
        go_2 *= -1
        update_color_for_circle()


def square_touch(x_new_):
    global go_1
    global go_2
    global Qq
    global color_square
    global color_triagle

    Qq = False

    if x_new_[0, 0] >= 236 or x_new_[0, 0] <= 20:
        Qq = True
        go_1 *= -1
        update_color_for_square()
    if x_new_[1, 0] >= 236 + 2 or x_new_[1, 0] <= 20 - 2:
        Qq = True
        go_2 *= -1
        update_color_for_square()

    if x_new_[0, 1] >= 236 or x_new_[0, 1] <= 20:
        Qq = True
        go_1 *= -1
        update_color_for_square()
    if x_new_[1, 1] >= 236 + 2 or x_new_[1, 1] <= 20 - 2:
        Qq = True
        go_2 *= -1
        update_color_for_square()

    if x_new_[0, 2] >= 236 or x_new_[0, 2] <= 20:
        Qq = True
        go_1 *= -1
        update_color_for_square()
    if x_new_[1, 2] >= 236 + 2 or x_new_[1, 2] <= 20 - 2:
        Qq = True
        go_2 *= -1
        update_color_for_square()


def check_collision_for_scaling(_ang, F):
    global scaling_num
    global Scale

    if int(ang) % 4 == 0:
        if F:
            for t in range(1, 4):
                if scaling_num > 2:
                    if scaling_num < 1:
                        continue
                    else:
                        scaling_num -= 0.01
                else:
                    scaling_num += 0.01

            Scale = scalingMatr(scaling_num)
    else:
        for t in range(1, 2):
            if scaling_num < 1:
                continue
            else:
                scaling_num -= 0.01

        if F:
            Scale = scalingMatr(scaling_num)


def check_prev_collision():
    global Qq
    global Qq_2

    if not Qq and not Qq_2:
        return True
    else:
        return False


def render():
    line(m, int(x_new[0, 0]), int(x_new[1, 0]), int(x_new[0, 1]), int(x_new[1, 1]), color_triagle)
    line(m, int(x_new[0, 1]), int(x_new[1, 1]), int(x_new[0, 2]), int(x_new[1, 2]), color_triagle)  # отрисовка треугольника
    line(m, int(x_new[0, 2]), int(x_new[1, 2]), int(x_new[0, 0]), int(x_new[1, 0]), color_triagle)

    line(m, int(sq1[0]), int(sq1[1]), int(sq2[0]), int(sq2[1]), color_square)
    line(m, int(sq1[0]), int(sq1[1]), int(sq3[0]), int(sq3[1]), color_square)  # отрисовка прямоугольника
    line(m, int(sq3[0]), int(sq3[1]), int(sq4[0]), int(sq4[1]), color_square)
    line(m, int(sq4[0]), int(sq4[1]), int(sq2[0]), int(sq2[1]), color_square)

    draw_circle(m, int(size / 2), int(size / 2), R, color_circle)  # отрисовка окружности


Scale = scalingMatr(scaling_num)

for i in range(N):
    ang = i * 3 * np.pi / N
    ang_2 = i * 6 * np.pi / N
    T = shiftMatr((-center[0], -center[1]))
    Rotation = rotMatr(ang)  # change rotation angle

    # Соприкосновение с окружностью
    circle_touch(x_new)

    U = check_prev_collision()

    # Соприкосновение с квадратом
    square_touch(x_new)

    check_collision_for_scaling(ang, U)

    # Изменение направления
    dx += go_1
    dy += go_2

    x_new = to_cart_coords(colculate_matrix(dx, dy, T_go_center, Scale, Rotation, T, x_proj))

    m = np.zeros((size, size, 3), dtype=np.uint8)

    render()

    im = plt.imshow(m)
    frames.append([im])

# gif animation creation
ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
writer = PillowWriter(fps=24)

plt.show()
ani.save("line.gif", writer=writer)