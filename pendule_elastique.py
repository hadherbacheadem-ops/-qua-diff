import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Paramètres (à remplacer par tes valeurs mesurées)
m = 0.2      # kg
k = 20.0     # N/m
g = 9.81     # m/s^2
l0 = 0.30    # longueur à vide du ressort (m)
l1 = 0.0     # m


def derivees(Y, t):
    r, rp, th, thp = Y
    rpp = r * thp**2 - (k / m) * (l1 + r - l0) + g * np.cos(th)
    thpp = (-g * np.sin(th) - 2 * rp * thp) / r
    return [rp, rpp, thp, thpp]


# Conditions initiales : r(0), r'(0), theta(0), theta'(0)
r_eq = l0 - l1 + m * g / k
Y0 = [r_eq, 0.0, np.radians(20), 0.0]
t = np.linspace(0, 10, 5000)

sol = odeint(derivees, Y0, t, rtol=1e-9, atol=1e-9)
r, rp, th, thp = sol.T

# Contrôle : énergie mécanique (doit rester ~constante)
E = 0.5 * m * (rp**2 + (r * thp)**2) - m * g * r * np.cos(th) + 0.5 * k * (l1 + r - l0)**2
print("Dérive relative de l'énergie :", (E.max() - E.min()) / abs(E[0]))

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, r); ax[0].set_ylabel("r (m)")
ax[1].plot(t, np.degrees(th)); ax[1].set_ylabel(r"$\theta$ (°)"); ax[1].set_xlabel("t (s)")
plt.tight_layout()
plt.savefig("pendule_elastique.png")
plt.show()
