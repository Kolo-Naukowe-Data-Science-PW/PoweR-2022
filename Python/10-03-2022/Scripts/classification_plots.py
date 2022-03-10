import matplotlib.pyplot as plt

# [ Data ]
# Płeć, Długość życia, Waga, Wzrost, Astma
data = [
    (1, 75, 104.2, 195, 0),
    (0, 87, 61.5, 172, 0),
    (0, 73, 54.8, 167, 1),
    (1, 56, 97.2, 182, 1),
    (0, 94, 59.3, 165, 0),
]


# [ Weight and height ]
x_range = [row[2] for row in data]  # weight
y_range = [row[3] for row in data]  # height

plt.xticks(range(int(min(x_range)), int(max(x_range)+1), 5))
plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("Weight and height - possible classification")
plt.plot(x_range, y_range, "o")

plt.show()


# [ Sex and asthma ]
x_range = [row[0] for row in data]  # sex
y_range = [row[-1] for row in data]  # asthma

plt.yticks(list(set(y_range)), ["No asthma", "Asthma"])
plt.xticks(list(set(x_range)), ["Woman", "Man"])
plt.xlabel("Sex")
plt.ylabel("Asthma")
plt.title("Sex and asthma - impossible classification")
plt.plot(x_range, y_range, "o")

plt.show()
