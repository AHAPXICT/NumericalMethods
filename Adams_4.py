import numpy as np


def f_function(x, y):
    # return x*x - 2*y
    return 1 + 2 * (y / x)


def y_correct_function(x):
    # return (3 / 4) * np.power(np.e, -2 * x) + 0.5 * x * x - 0.5 * x + 0.25
    return x * x - x


a = float(input("Enter a: "))
b = float(input("Enter b: "))
n = int(input("Enter n: "))
h = (b - a) / n

x = [None for i in range(n + 1)]
y = [None for i in range(n + 1)]
f = [None for i in range(n + 1)]

print('If u wonna use R-K press 1')
print('If u wonna use Euler press 2')
print('If u wonna writte correct values press 3')

key = int(input())

if (key == 1):
    x[0] = a
    y[0] = y_correct_function(a)
    f[0] = f_function(x[0], y[0])

    for i in range(0, 3):
        x[i + 1] = x[i] + h
        k1 = h * f[i]
        k2 = h * f_function(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f_function(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f_function(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[i + 1] = f_function(x[i + 1], y[i + 1])
elif (key == 2):
    x[0] = a
    y[0] = y_correct_function(a)
    f[0] = f_function(x[0], y[0])

    for i in range(0, 3):
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h * f_function(x[i], y[i])
        f[i + 1] = f_function(x[i + 1], y[i + 1])

elif (key == 3):

    x[0] = a
    y[0] = y_correct_function(a)
    f[0] = f_function(x[0], y[0])

    for i in range(0, 3):
        x[i + 1] = x[i] + h
        print("Enter y[" + str(i + 1) + "]: ")
        y[i + 1] = float(input())
        f[i + 1] = f_function(x[i + 1], y[i + 1])

error = 0.0

for i in range(3, n):
    x[i + 1] = x[i] + h
    y[i + 1] = y[i] + (h / 24) * (55 * f[i] - 59 * f[i - 1] + 37 * f[i - 2] - 9 * f[i - 3])
    f[i + 1] = f_function(x[i + 1], y[i + 1])

    if (np.abs(y[i + 1] - y_correct_function(x[i + 1])) > error):
        error = np.abs(y[i + 1] - y_correct_function(x[i + 1]))

for i in range(0, n + 1):
    print("I = ", i)
    print("X = ", x[i])
    print("Y = ", y[i])
    print("==========================================================")

print("Error: ", error)
