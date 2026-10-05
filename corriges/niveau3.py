"""Niveau 3 : oscillateur harmonique  x'' = -omega^2 x
Passage d'une équation d'ordre 2 à un système d'ordre 1, résolu avec
solve_ivp puis avec Euler, et contrôle par l'énergie."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- Paramètres ----------
omega = 1.0              # pulsation propre (rad/s)
x0 = 1.0                 # position initiale
v0 = 0.0                 # vitesse initiale
T0 = 2 * np.pi / omega   # période théorique
t_fin = 10 * T0          # on simule 10 périodes


# ---------- 1. Le système d'ordre 1 ----------
# Y = [x, v]  et  dY/dt = [v, -omega^2 x]
def f(t, Y):
    x, v = Y
    dx_dt = v
    dv_dt = -omega**2 * x
    return [dx_dt, dv_dt]


# ---------- 2. Résolution avec solve_ivp ----------
t = np.linspace(0, t_fin, 2000)
sol = solve_ivp(f, (0, t_fin), [x0, v0], t_eval=t, rtol=1e-9, atol=1e-9)
x = sol.y[0]
v = sol.y[1]
E = 0.5 * v**2 + 0.5 * omega**2 * x**2


# ---------- 3. Même système avec Euler (fait à la main) ----------
dt = 0.01
n = int(round(t_fin / dt))
t_e = np.linspace(0, n * dt, n + 1)
x_e = np.zeros(n + 1)
v_e = np.zeros(n + 1)
x_e[0] = x0
v_e[0] = v0
for i in range(n):
    dx_dt, dv_dt = f(t_e[i], [x_e[i], v_e[i]])   # on réutilise la même f
    x_e[i + 1] = x_e[i] + dt * dx_dt
    v_e[i + 1] = v_e[i] + dt * dv_dt
E_e = 0.5 * v_e**2 + 0.5 * omega**2 * x_e**2


# ---------- 4. Vérifications chiffrées ----------
x_exact = x0 * np.cos(omega * t)
print("Écart max solve_ivp / solution exacte :", np.max(np.abs(x - x_exact)))
print("solve_ivp : E_fin / E_0 =", E[-1] / E[0])
print("Euler     : E_fin / E_0 =", E_e[-1] / E_e[0])


# ---------- 5. Graphiques ----------
plt.figure()
plt.plot(t, x_exact, "k--", label="exacte : $x_0\\cos(\\omega t)$")
plt.plot(t, x, label="solve_ivp")
plt.plot(t_e, x_e, label="Euler, dt = 0.01", alpha=0.7)
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.legend()

plt.figure()
plt.plot(t, E, label="solve_ivp")
plt.plot(t_e, E_e, label="Euler, dt = 0.01")
plt.xlabel("t (s)")
plt.ylabel("E (J/kg)")
plt.legend()

plt.show()
