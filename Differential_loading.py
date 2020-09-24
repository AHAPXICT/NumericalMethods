import numpy as np


def q_x(x):
    # return 2
    return 3


def p_x(x):
    # return x + 1
    return 2


def f_x(x):
    return -6 * (x * x) + 8 * x + 1


def Z1_function(x, z1):
    return -(z1 * z1) - p_x(x) * z1 + q_x(x)


def Z2_function(x, z1, z2):
    return -z2 * (z1 + p_x(x)) + f_x(x)


def y_function_z(Z1, Z2, y):
    return Z1 * y + Z2


def U1_function(x, u1):
    return -(u1 * u1) * q_x(x) + u1 * p_x(x) + 1


def U2_function(x, u1, u2):
    return -u1 * (u2 * q_x(x) + f_x(x))


def y_function_u(u1, u2, y):
    return u1 * y + u2


def y_correct_function(x):
    return 2 * x * x + 1


print("Enter values for equal: ")
alpha_0 = 0  # float(input("Enter alpha 0: "))
betta_0 = -1  # -1#float(input("Enter betta 0: "))
alpha_1 = 0  # 1#float(input("Enter alpha 1: "))
betta_1 = 1  # 0#float(input("Enter betta 1: "))
hamma_0 = 0  # -1#float(input("Enter hamma 0: "))
hamma_1 = 4  # 4#float(input("Enter hamma 1: "))

a = float(input("Enter start: "))
b = float(input("Enter end: "))
n = int(input("Enter n: "))
h = (b - a) / n

x = [None for i in range(n + 1)]
y = [None for i in range(n + 1)]
correct_y = [None for i in range(n + 1)]

if (betta_0 != 0):
    Z1 = [None for i in range(n + 1)]
    Z2 = [None for i in range(n + 1)]

    x[0] = a
    Z1[0] = -alpha_0 / betta_0
    Z2[0] = hamma_0 / betta_0

    for i in range(0, n):
        x[i + 1] = x[i] + h

        delta_z1 = h * Z1_function(x[i] + h / 2, Z1[i] + h / 2 * Z1_function(x[i], Z1[i]))
        Z1[i + 1] = Z1[i] + delta_z1

        delta_z2 = h * Z2_function(x[i] + h / 2, Z1[i], Z2[i] + h / 2 * Z2_function(x[i], Z1[i], Z2[i]))
        Z2[i + 1] = Z2[i] + delta_z2

    y[n] = (hamma_1 - betta_1 * Z2[n]) / (alpha_1 + betta_1 * Z1[n])
    print(y[n])
    for i in range(n, 0, -1):
        delta_y = h * y_function_z(Z1[i], Z2[i], y[i] - h / 2 * y_function_z(Z1[i], Z2[i], y[i]))
        y[i - 1] = y[i] - delta_y

    for i in range(0, n + 1):
        print("I = ", i)
        print("Z1 = ", Z1[i])
        print("Z2 = ", Z2[i])
        print("Y = ", y[i])
        print("==========================================================")

elif (alpha_0 != 0):
    U1 = [None for i in range(n + 1)]
    U2 = [None for i in range(n + 1)]

    x[0] = a
    U1[0] = -betta_0 / alpha_0
    U2[0] = hamma_0 / alpha_0

    for i in range(0, n):
        x[i + 1] = x[i] + h

        delta_u1 = h * U1_function(x[i] + h / 2, U1[i] + h / 2 * U1_function(x[i], U1[i]))
        U1[i + 1] = U1[i] + delta_u1

        delta_u2 = h * U2_function(x[i] + h / 2, U1[i], U2[i] + h / 2 * U2_function(x[i], U1[i], U2[i]))
        U2[i + 1] = U2[i] + delta_u2

    y[n] = (hamma_1 * U1[n] + betta_1 * U2[n]) / (betta_1 + alpha_1 * U1[n])
    for i in range(n, 0, -1):
        delta_y = h * y_function_u(U1[i], U2[i], y[i] + h / 2 * y_function_u(U1[i], U2[i], y[i]))
        y[i - 1] = y[i] - delta_y

    for i in range(0, n + 1):
        print("I = ", i)
        print("U1 = ", U1[i])
        print("U2 = ", U2[i])
        print("Y = ", y[i])
        print("==========================================================")

error = 0

for i in range(0, n):
    if (np.abs(y[i] - y_correct_function(x[i])) > error):
        error = np.abs(y[i] - y_correct_function(x[i]))

print("Error: ", error)
