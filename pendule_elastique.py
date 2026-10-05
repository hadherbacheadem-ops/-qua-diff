"""Chute d'un grimpeur retenue par une corde élastique passant dans une dégaine.

Phase 1 : chute libre verticale depuis M1 (distance l au-dessus de O, angle theta0).
Phase 2 : la corde se tend au point symétrique sous O ; pendule élastique en
coordonnées polaires (r = OM, alpha mesuré depuis la verticale descendante).

Hypothèses : dégaine sans frottement (tension uniforme), assureur fixe à la
distance l1 de O, corde à sa longueur naturelle l0 = l1 + l à l'instant où
elle se tend, corde incapable de pousser (T >= 0), paroi ignorée.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Paramètres (ordres de grandeur réalistes, à remplacer par tes valeurs)
m = 80.0                 # masse du grimpeur (kg)
g = 9.81                 # m/s^2
l = 2.0                  # longueur de corde au-dessus de la dégaine (m)
l1 = 10.0                # distance dégaine -> assureur (m)
l0 = l1 + l              # longueur naturelle de la corde utilisée (m)
EA = 22e3                # rigidité linéique de la corde (N), ~20-25 kN pour une corde dynamique
k = EA / l0              # raideur de toute la corde : dépend de la longueur sortie
c = 0.0                  # amortissement visqueux de la corde (N.s/m), 0 = élastique pur
theta0 = np.radians(20)  # angle initial de M1 par rapport à la verticale montante


def tension(r, rp):
    allongement = l1 + r - l0
    if allongement <= 0:
        return 0.0                       # corde détendue
    return max(0.0, k * allongement + c * rp)


def derivees(Y, t):
    r, rp, a, ap = Y
    T = tension(r, rp)
    rpp = r * ap**2 - T / m + g * np.cos(a)
    app = (-g * np.sin(a) - 2 * rp * ap) / r
    return [rp, rpp, ap, app]


def conditions_initiales(theta):
    """Fin de la chute libre : chute verticale de 2 l cos(theta)."""
    vi = np.sqrt(4 * g * l * np.cos(theta))
    # vitesse (0, -vi) projetée sur e_r = (sin a, -cos a) et e_a = (cos a, sin a)
    return [l, vi * np.cos(theta), theta, -vi * np.sin(theta) / l]


def simuler(theta, t):
    sol = odeint(derivees, conditions_initiales(theta), t, rtol=1e-10, atol=1e-10)
    r, rp, a, ap = sol.T
    T = np.array([tension(ri, rpi) for ri, rpi in zip(r, rp)])
    return r, rp, a, ap, T


if __name__ == "__main__":
    t = np.linspace(0, 6, 60001)

    # Validation : chute verticale (theta = 0) contre la formule de la force de choc
    *_, T_vert = simuler(0.0, t)
    f = 2 * l / l0                                       # facteur de chute
    T_th = m * g + np.sqrt((m * g)**2 + 2 * m * g * EA * f)
    print(f"theta=0  : T_max simulée = {T_vert.max():.1f} N, formule = {T_th:.1f} N")

    r, rp, a, ap, T = simuler(theta0, t)
    if c == 0:
        E = (0.5 * m * (rp**2 + (r * ap)**2) - m * g * r * np.cos(a)
             + 0.5 * k * np.clip(l1 + r - l0, 0, None)**2)
        print(f"Dérive relative de l'énergie : {np.ptp(E) / abs(E[0]):.1e}")
    i = T.argmax()
    print(f"theta0={np.degrees(theta0):.0f}° : T_max = {T[i]:.0f} N ({T[i] / (m * g):.1f} g) "
          f"à t = {t[i]:.3f} s, allongement max = {(l1 + r - l0).max():.2f} m")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    ax[0].plot(t, r); ax[0].set_ylabel("r = OM (m)")
    ax[1].plot(t, np.degrees(a)); ax[1].set_ylabel(r"$\alpha$ (°)")
    ax[2].plot(t, T / 1e3); ax[2].set_ylabel("T (kN)"); ax[2].set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig("pendule_elastique.png")
    plt.show()
