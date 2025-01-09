import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
boltz = 5.67e-8

# Temperatures
T_inf = 25 + 273.15
T_init = 25 + 273.15
T_wall = T_inf

# Sizing variables
batt = {
    "along_height": 5,
    "along_length": 7,
    "R": 15e-3,
    "V": 4,
    "D": 21.55e-3,
    "len": 70.15e-3,  # Battery height
    "As": np.pi * 21.55e-3 * 70.15e-3,
    "I": 225 / 5,  # Current per height
}
batt["q"] = batt["R"] * batt["I"]**2

Sl = 1  # Spacing lengthwise between batteries
St = 1  # Spacing widthwise between batteries
L = 8 * Sl
H = 7 * St

# Properties of air
air = {
    "rho": 1.1614,
    "v": 15.89e-6,
    "k": 26.3e-3,
    "Pr": 0.707,
    "maxPr": 0.7,
    "cp": 1.007,
}

flow_rate = 0.075  # m^3/sec
Press_max = 65  # Pa
T_max = 75  # Celsius
epsilon = 0.8

# Flow variables
f = 1
dp = 65
x = 0.5

V_avrg = (St - batt["D"]) / St * np.sqrt(2 * dp / (air["rho"] * batt["along_length"] * x * f))
Re = V_avrg * batt["D"] / air["v"]

air["mdot"] = V_avrg * air["rho"] * H * L

# Air temperature array
T_air = np.ones(batt["along_length"]) * T_init
for j in range(1, batt["along_length"]):
    T_air[j] = T_air[j - 1] + batt["q"] * batt["along_height"] / (air["mdot"] * air["cp"])

# Initial temperature matrix
T_init_matrix = np.full((batt["along_height"], batt["along_length"]), T_init)

# Function to solve
def func(T, T_air, boltz, T_wall, epsilon, batt, St, Sl, air):
    T = T.reshape(batt["along_height"], batt["along_length"])
    eq = np.zeros_like(T)

    V_avrg = (St - batt["D"]) / St * np.sqrt(2 * dp / (air["rho"] * batt["along_length"] * x * f))
    Re = V_avrg * batt["D"] / air["v"]

    if 10 < Re < 1e2:
        C1, m = 0.8, 0.4
    elif 1e3 < Re < 2e5:
        C1, m = 0.27, 0.63
    else:
        C1, m = 0.021, 0.84

    C2 = 0.95
    Nu = C2 * C1 * Re**m * air["Pr"]**0.36 * (air["Pr"] / 0.7035)**0.25
    h = Nu * air["k"] / batt["D"]

    for row in range(batt["along_height"]):
        for col in range(batt["along_length"]):
            if row == 0:
                if col == 0:  # Top-left corner
                    T_avrg = ((T[row + 1, col]**4 + T[row, col + 1]**4) / 2)**0.25
                elif col == batt["along_length"] - 1:  # Top-right corner
                    T_avrg = ((T[row + 1, col]**4 + T[row, col - 1]**4) / 2)**0.25
                else:  # Top edge
                    T_avrg = ((T[row + 1, col]**4 + T[row, col - 1]**4 + T[row, col + 1]**4) / 3)**0.25
            elif row == batt["along_height"] - 1:
                if col == 0:  # Bottom-left corner
                    T_avrg = ((T[row - 1, col]**4 + T[row, col + 1]**4) / 2)**0.25
                elif col == batt["along_length"] - 1:  # Bottom-right corner
                    T_avrg = ((T[row - 1, col]**4 + T[row, col - 1]**4) / 2)**0.25
                else:  # Bottom edge
                    T_avrg = ((T[row - 1, col]**4 + T[row, col - 1]**4 + T[row, col + 1]**4) / 3)**0.25
            else:
                if col == 0:  # Left edge
                    T_avrg = ((T[row - 1, col]**4 + T[row + 1, col]**4 + T[row, col + 1]**4) / 3)**0.25
                elif col == batt["along_length"] - 1:  # Right edge
                    T_avrg = ((T[row - 1, col]**4 + T[row + 1, col]**4 + T[row, col - 1]**4) / 3)**0.25
                else:  # Center
                    T_avrg = ((T[row - 1, col]**4 + T[row + 1, col]**4 + T[row, col - 1]**4 + T[row, col + 1]**4) / 4)**0.25

            eq[row, col] = (
                batt["I"]**2 * batt["R"]
                - h * batt["As"] * (T[row, col] - T_air[col])
                - epsilon * boltz * batt["As"] * (T[row, col]**4 - T_avrg**4)
            )

    return eq.flatten()

# Solve using fsolve
T_solution = fsolve(func, T_init_matrix.flatten(), args=(T_air, boltz, T_wall, epsilon, batt, St, Sl, air))
T_matrix = T_solution.reshape(batt["along_height"], batt["along_length"]) - 273.15

# Display results
print("Temperature matrix (in Celsius):")
print(T_matrix)

# Plot the temperature matrix
plt.imshow(T_matrix, cmap="jet", origin="lower")
plt.colorbar(label="Temperature (C)")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.title("Temperature Distribution")
plt.show()
