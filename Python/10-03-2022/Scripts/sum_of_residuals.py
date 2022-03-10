import matplotlib.pyplot as plt
from math import sin, pi

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Sum of residuals = 0')

ax1.plot([], [], '-', color="black", label="Residual")

x_range = [pi/2, pi*(3/2), pi*(5/2), pi*(7/2)]

ax1.scatter(x_range,
            [sin(x) for x in x_range],
            color="blue",
            label="Values")

ax1.plot([x/10 for x in range(121)],
         [sin(x/10) for x in range(121)],
         color="red",
         label="Regression values")

ax2.scatter(x_range,
            [sin(x-pi) for x in x_range],
            color="blue")

ax2.plot([x/10 for x in range(121)],
         [sin(x/10) for x in range(121)],
         color="red")

ax3.scatter(x_range,
            [sin(x-pi) for x in x_range],
            color="blue")

ax3.plot([x/10 for x in range(121)],
         [0 for x in range(121)],
         color="red")

for x in x_range:
    ax1.plot([x, x], [sin(x), sin(x)], color="black")
    ax2.plot([x, x], [sin(x), sin(x-pi)], color="black")
    ax3.plot([x, x], [sin(x-pi), 0], color="black")


fig.legend()
plt.show()
