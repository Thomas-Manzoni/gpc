"""Simulates the stick output of rhythm_shot_MP_upt_v2.gpc and plots it.

Requires: pip install matplotlib
"""
import random

import matplotlib.pyplot as plt

# Same values as the GPC script (defaults)
TOTAL_TIME = 555
RISE_TIME = 75
LATENCY_ON = True
LATENCY_VAL = 55
UP_HOLD_TIME = 100
X_NOISE = 2
X_PATH_MAX = 30
STICK_DOWN = 100
STICK_UP = -100

FRAME_MS = 10  # approx get_rtime() on the Zen
SHOTS = 5


def simulate_shot(hold_time_base):
    """Returns lists of (t, RX, RY) for one shot, mirroring the GPC state machine."""
    hold_time = hold_time_base + (LATENCY_VAL if LATENCY_ON else 0)
    state, timer, t = "HOLD_DOWN", 0, 0
    x_start = random.randint(0, X_PATH_MAX * 2) - X_PATH_MAX
    x_end = random.randint(0, X_PATH_MAX * 2) - X_PATH_MAX
    y_start = STICK_DOWN - (x_start * x_start) // 200
    y_end = STICK_UP + (x_end * x_end) // 200
    ts, rxs, rys = [], [], []

    while state != "IDLE":
        timer += FRAME_MS
        ry = 0
        x_val = x_start

        if state == "HOLD_DOWN":
            ry = y_start
            if timer >= hold_time:
                timer, state = 0, "PULL_UP"
        elif state == "PULL_UP":
            if timer >= RISE_TIME:
                ry = y_end
                x_val = x_end
                timer, state = 0, "HOLD_UP"
            else:
                frac = (timer * 100) // RISE_TIME
                ry = y_start + int((y_end - y_start) * frac / 100)
                x_val = x_start + int((x_end - x_start) * timer / RISE_TIME)
        else:
            ry = y_end
            x_val = x_end
            if timer >= UP_HOLD_TIME:
                timer, state = 0, "IDLE"

        rx = x_val + random.randint(0, X_NOISE * 2) - X_NOISE
        ts.append(t)
        rxs.append(rx)
        rys.append(ry)
        t += FRAME_MS

    return ts, rxs, rys


fig, (ax_t, ax_xy) = plt.subplots(1, 2, figsize=(12, 5))

for i in range(SHOTS):
    ts, rxs, rys = simulate_shot(TOTAL_TIME)
    ax_t.plot(ts, rys, color="tab:blue", alpha=0.6, label="RY" if i == 0 else None)
    ax_t.plot(ts, rxs, color="tab:orange", alpha=0.6, label="RX" if i == 0 else None)
    # flip Y so "down" is at the bottom, like the physical stick
    ax_xy.plot(rxs, [-y for y in rys], marker=".", alpha=0.6)

ax_t.set_title("Stick values over time")
ax_t.set_xlabel("ms")
ax_t.set_ylabel("value")
ax_t.legend()
ax_t.grid(True)

ax_xy.set_title(f"Stick path ({SHOTS} shots, bottom = down)")
ax_xy.set_xlabel("RX")
ax_xy.set_ylabel("RY (up is positive here)")
ax_xy.set_xlim(-100, 100)
ax_xy.set_ylim(-110, 110)
ax_xy.set_aspect("equal")
ax_xy.grid(True)

plt.tight_layout()
plt.show()
