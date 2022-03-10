from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# [ Data ]
# Year, country's debt
data = [
     (2002, 338),
     (2003, 394),
     (2004, 420),
     (2005, 461),
     (2006, 505),
     (2007, 528),
     (2008, 600),
]


# [ Year and debt ]
x_range = [row[0] for row in data]  # year
y_range = [row[1] for row in data]  # debt

np_x_range = np.array(x_range).reshape((-1, 1))
np_y_range = np.array(y_range)

reg = LinearRegression().fit(np_x_range, np_y_range)

a = reg.intercept_
b = reg.coef_
y_predicted = [a + b*x for x in np_x_range]

plt.xticks(np_x_range)
plt.yticks(np_y_range)
plt.xlabel("Year")
plt.ylabel("Debt")
plt.title("The country's debt in 2002 - 2008")
plt.plot(np_x_range, np_y_range, "o", label="real debt")
plt.plot(np_x_range, y_predicted,
         label="predict debt",
         color="gray")


# [ residuals ]

plt.plot((x_range[0], x_range[0]),
         (y_predicted[0], y_range[0]),
         color="black",
         label="residuals")

for index, x in enumerate(x_range[1:]):
    plt.plot([x, x], [y_predicted[index+1], y_range[index+1]], color="black")

plt.legend()
plt.show()

residuals_sum = float(sum(y_predicted) - sum(np_y_range))
print("Residuals sum:", residuals_sum, "~", round(residuals_sum, 2))
