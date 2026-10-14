"""Reads the controller output of the Cronus Zen and live-plots the right stick.

Plug the Zen into the PC (Zen OUTPUT port -> PC) with your controller in the
Zen's input port. Windows then sees the Zen as an Xbox controller, so whatever
the running GPC script sends (including the rhythm-shot stick motion) shows up
here.

Each shot is timed against rhythm_shot_MP_upt_v5.gpc: how long RY stays parked
at the bottom (hold_time = TOTAL_TIME_MP + latency_val) and how long the ramp to
the top takes (RISE_TIME_MP).

    pip install matplotlib
    python plot_rs.py                  # live plot + shot timings
    python plot_rs.py --list           # what Windows can see right now
    python plot_rs.py --monitor        # text dump of every axis (find mappings)
    python plot_rs.py --record run.csv # live plot + save samples
    python plot_rs.py --replay run.csv # re-time and plot a saved run

Values are scaled to -100..100 and RY uses the GPC convention: positive = stick
DOWN, negative = stick UP (matches STICK_DOWN / STICK_UP in the .gpc scripts).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import threading
import time
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

SAMPLE_HZ = 250
STICK_SCALE = 32767 / 100.0


class NoDeviceError(RuntimeError):
    """No controller is currently exposing stick data."""


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
class XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", wintypes.WORD),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD), ("Gamepad", XInputGamepad)]


class XInputReader:
    """Reads the Zen's emulated pad through the Windows XInput API."""

    name = "xinput"

    def __init__(self, pad: int | None = None):
        self._dll = None
        for lib in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
            try:
                self._dll = ctypes.windll.LoadLibrary(lib)
                break
            except OSError:
                continue
        if self._dll is None:
            raise RuntimeError("XInput is not available (Windows only).")

        self._state = XInputState()
        self.pad = pad if pad is not None else self._find_pad()

    def _connected(self, pad: int) -> bool:
        return self._dll.XInputGetState(pad, ctypes.byref(self._state)) == 0

    def _find_pad(self) -> int:
        for pad in range(4):
            if self._connected(pad):
                return pad
        raise NoDeviceError("no XInput pad in slots 0-3")

    def read(self) -> tuple[float, float]:
        if self._dll.XInputGetState(self.pad, ctypes.byref(self._state)) != 0:
            return 0.0, 0.0
        gp = self._state.Gamepad
        # XInput Y is positive-up; GPC is positive-down.
        return gp.sThumbRX / STICK_SCALE, -gp.sThumbRY / STICK_SCALE

    def axes(self) -> dict[str, float]:
        self._dll.XInputGetState(self.pad, ctypes.byref(self._state))
        gp = self._state.Gamepad
        return {
            "LX": gp.sThumbLX / STICK_SCALE,
            "LY": gp.sThumbLY / STICK_SCALE,
            "RX": gp.sThumbRX / STICK_SCALE,
            "RY": gp.sThumbRY / STICK_SCALE,
            "LT": gp.bLeftTrigger / 2.55,
            "RT": gp.bRightTrigger / 2.55,
        }


class JoyInfoEx(ctypes.Structure):
    _fields_ = [(n, wintypes.DWORD) for n in (
        "dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos", "dwUpos",
        "dwVpos", "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1", "dwReserved2",
    )]


class WinmmReader:
    """Legacy joystick API - picks up pads the XInput driver doesn't claim.

    Useful when the Zen outputs as DS4/Switch rather than Xbox.
    """

    name = "winmm"
    AXIS_FIELDS = {"x": "dwXpos", "y": "dwYpos", "z": "dwZpos",
                   "r": "dwRpos", "u": "dwUpos", "v": "dwVpos"}

    def __init__(self, pad: int | None = None, axes: tuple[str, str] = ("u", "r")):
        self._dll = ctypes.windll.winmm
        self._info = JoyInfoEx()
        self._info.dwSize = ctypes.sizeof(JoyInfoEx)
        self._info.dwFlags = 0xFF  # JOY_RETURNALL
        self.pad = pad if pad is not None else self._find_pad()
        self._fx = self.AXIS_FIELDS[axes[0].lower()]
        self._fy = self.AXIS_FIELDS[axes[1].lower()]

    def _connected(self, pad: int) -> bool:
        return self._dll.joyGetPosEx(pad, ctypes.byref(self._info)) == 0

    def _find_pad(self) -> int:
        for pad in range(self._dll.joyGetNumDevs()):
            if self._connected(pad):
                return pad
        raise NoDeviceError("no winmm joystick responding")

    @staticmethod
    def _scale(raw: int) -> float:
        return (raw - 32767) / 327.67  # 0..65535 -> -100..100

    def read(self) -> tuple[float, float]:
        if not self._connected(self.pad):
            return 0.0, 0.0
        return self._scale(getattr(self._info, self._fx)), self._scale(getattr(self._info, self._fy))

    def axes(self) -> dict[str, float]:
        self._connected(self.pad)
        return {k.upper(): self._scale(getattr(self._info, f))
                for k, f in self.AXIS_FIELDS.items()}


class PygameReader:
    """Optional backend; handles exotic pads the two stdlib APIs miss."""

    name = "pygame"

    def __init__(self, pad: int | None = None, axes: tuple[str, str] = ("2", "3")):
        try:
            import pygame  # imported lazily so the default path needs no deps
        except ImportError as exc:
            raise NoDeviceError("pygame is not installed (pip install pygame)") from exc

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise NoDeviceError("pygame found no joysticks")
        self.pad = pad or 0
        self._js = pygame.joystick.Joystick(self.pad)
        self._js.init()
        self._ax_x, self._ax_y = int(axes[0]), int(axes[1])
        print(f"pygame: {self._js.get_name()} ({self._js.get_numaxes()} axes)")

    def read(self) -> tuple[float, float]:
        self._pygame.event.pump()
        # pygame already reports Y positive-down.
        return self._js.get_axis(self._ax_x) * 100.0, self._js.get_axis(self._ax_y) * 100.0

    def axes(self) -> dict[str, float]:
        self._pygame.event.pump()
        return {str(i): self._js.get_axis(i) * 100.0 for i in range(self._js.get_numaxes())}


BACKENDS = {"xinput": XInputReader, "winmm": WinmmReader, "pygame": PygameReader}

NO_DEVICE_HELP = """
No controller is exposing stick data. Checklist:
  1. Zen OUTPUT port -> PC, controller in the Zen's INPUT port (not the other
     way round), and the controller powered ON - a pad that is only charging
     enumerates but never creates a gamepad device.
  2. A script must be loaded and running on the Zen (Zen Studio -> Program).
  3. The Zen's output protocol must match the backend: Xbox -> xinput,
     DS4/Switch -> try --backend winmm.
  4. Run `python plot_rs.py --list` to see what Windows reports.
"""


def _try(backend: str, pad: int | None, axes: tuple[str, str]):
    cls = BACKENDS[backend]
    return cls(pad, axes) if cls is not XInputReader else cls(pad)


def make_reader(backend: str, pad: int | None, axes: tuple[str, str], wait: bool):
    order = [backend] if backend != "auto" else ["xinput", "winmm"]
    deadline_msg_shown = False
    while True:
        problems = []
        for name in order:
            try:
                return _try(name, pad, axes)
            except NoDeviceError as exc:
                problems.append(f"  {name}: {exc}")
        if not wait:
            raise SystemExit("\n".join(problems) + "\n" + NO_DEVICE_HELP)
        if not deadline_msg_shown:
            print("\n".join(problems))
            print("Waiting for a controller... (Ctrl+C to quit)")
            deadline_msg_shown = True
        time.sleep(0.5)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def list_devices() -> None:
    print("XInput slots:")
    try:
        dll = ctypes.windll.LoadLibrary("xinput1_4")
        state = XInputState()
        for pad in range(4):
            ok = dll.XInputGetState(pad, ctypes.byref(state)) == 0
            print(f"  {pad}: {'CONNECTED' if ok else '-'}")
    except OSError as exc:
        print(f"  unavailable: {exc}")

    print("winmm joysticks:")
    winmm = ctypes.windll.winmm
    info = JoyInfoEx()
    info.dwSize = ctypes.sizeof(JoyInfoEx)
    info.dwFlags = 0xFF
    found = False
    for pad in range(winmm.joyGetNumDevs()):
        if winmm.joyGetPosEx(pad, ctypes.byref(info)) == 0:
            print(f"  {pad}: CONNECTED (x={info.dwXpos} y={info.dwYpos} "
                  f"z={info.dwZpos} r={info.dwRpos} u={info.dwUpos} v={info.dwVpos})")
            found = True
    if not found:
        print(f"  none responding (of {winmm.joyGetNumDevs()} slots)")

    print("pygame:")
    try:
        import pygame

        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        for i in range(count):
            js = pygame.joystick.Joystick(i)
            print(f"  {i}: {js.get_name()} ({js.get_numaxes()} axes)")
        if not count:
            print("  no joysticks")
    except ImportError:
        print("  not installed (optional)")

    print(NO_DEVICE_HELP)


def monitor(reader) -> None:
    """Prints every axis so you can work out which one is the right stick."""
    print(f"Monitoring {reader.name} pad {reader.pad}. Move the right stick. Ctrl+C to quit.")
    try:
        while True:
            values = reader.axes()
            print("  ".join(f"{k}={v:7.1f}" for k, v in values.items()), end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


# --------------------------------------------------------------------------- #
# Shot analysis
# --------------------------------------------------------------------------- #
# v5 parks RY at y_start = 100 - x_start^2/200 (>= 98 for X_PATH_MAX 15), ramps
# linearly to y_end = -100 + x_end^2/200 over RISE_TIME_MP, then holds UP_HOLD_TIME.
DOWN_LEVEL = 85.0  # RY at/above this = stick parked at the bottom
UP_LEVEL = -85.0  # RY at/below this = stick parked at the top
RISE_TIMEOUT_MS = 600.0  # a "rise" slower than this is a manual stick move, not a shot

# v5 defaults: TOTAL_TIME_DEFAULT 563 + latency_val 55, RISE_TIME_DEFAULT 100.
EXPECT_HOLD_MS = 618.0
EXPECT_RISE_MS = 100.0


@dataclass
class Shot:
    index: int
    t_start: float
    hold_ms: float
    rise_ms: float
    up_hold_ms: float
    y_bottom: float
    y_top: float
    x_start: float
    x_end: float
    t_rise_start: float
    t_rise_end: float
    t_end: float

    @property
    def total_ms(self) -> float:
        return self.hold_ms + self.rise_ms + self.up_hold_ms

    def describe(self, expect_hold: float | None, expect_rise: float | None) -> str:
        hold = f"hold {self.hold_ms:7.1f}"
        rise = f"rise {self.rise_ms:6.1f}"
        if expect_hold:
            hold += f" ({self.hold_ms - expect_hold:+6.1f})"
        if expect_rise:
            rise += f" ({self.rise_ms - expect_rise:+6.1f})"
        return (f"#{self.index:<3} {hold}  {rise}  up-hold {self.up_hold_ms:6.1f}  "
                f"total {self.total_ms:7.1f} ms  x {self.x_start:+5.1f} -> {self.x_end:+5.1f}")


class ShotAnalyzer:
    """Streaming detector: measures the bottom hold and the rise of each shot.

    Phase edges come from a least-squares fit of the linear ramp extrapolated out
    to the two plateau levels, so the numbers don't depend on where the sampler
    happened to land relative to the threshold crossings.
    """

    def __init__(self, expect_hold: float | None = None, expect_rise: float | None = None):
        self.shots: list[Shot] = []
        self.expect_hold = expect_hold
        self.expect_rise = expect_rise
        self._reset()

    def _reset(self) -> None:
        self._state = "IDLE"
        self._t_down = 0.0
        self._bottom_y: list[float] = []
        self._bottom_x: list[float] = []
        self._top_y: list[float] = []
        self._top_x: list[float] = []
        self._ramp: list[tuple[float, float]] = []
        self._t_top = 0.0

    def feed(self, t: float, rx: float, ry: float) -> Shot | None:
        if self._state == "IDLE":
            if ry >= DOWN_LEVEL:
                self._state = "DOWN"
                self._t_down = t
                self._bottom_y, self._bottom_x = [ry], [rx]
        elif self._state == "DOWN":
            if ry >= DOWN_LEVEL:
                self._bottom_y.append(ry)
                self._bottom_x.append(rx)
            else:
                self._state = "RISE"
                self._ramp = [(t, ry)]
        elif self._state == "RISE":
            if ry <= UP_LEVEL:
                self._state = "UP"
                self._t_top = t
                self._top_y, self._top_x = [ry], [rx]
            elif t - self._ramp[0][0] > RISE_TIMEOUT_MS:
                self._reset()  # stick wandered; not a scripted shot
            else:
                self._ramp.append((t, ry))
        elif self._state == "UP":
            if ry <= UP_LEVEL:
                self._top_y.append(ry)
                self._top_x.append(rx)
            else:
                shot = self._finish(t)
                self._reset()
                return shot
        return None

    def _finish(self, t_end: float) -> Shot:
        y_bottom = median(self._bottom_y)
        y_top = median(self._top_y)
        t_leave, t_reach = self._ramp_edges(y_bottom, y_top)
        shot = Shot(
            index=len(self.shots) + 1,
            t_start=self._t_down,
            hold_ms=t_leave - self._t_down,
            rise_ms=t_reach - t_leave,
            up_hold_ms=t_end - t_reach,
            y_bottom=y_bottom,
            y_top=y_top,
            x_start=median(self._bottom_x),
            x_end=median(self._top_x),
            t_rise_start=t_leave,
            t_rise_end=t_reach,
            t_end=t_end,
        )
        self.shots.append(shot)
        return shot

    def _ramp_edges(self, y_bottom: float, y_top: float) -> tuple[float, float]:
        """Times at which the ramp leaves y_bottom and reaches y_top."""
        if len(self._ramp) >= 2:
            ts = [p[0] for p in self._ramp]
            ys = [p[1] for p in self._ramp]
            slope, intercept = np.polyfit(ts, ys, 1)
            if slope < -1e-6:
                return (y_bottom - intercept) / slope, (y_top - intercept) / slope
        # too few ramp samples to fit: fall back to the raw threshold crossings
        return (self._ramp[0][0] if self._ramp else self._t_top), self._t_top

    def summary(self) -> str:
        if not self.shots:
            return "no shots yet"
        holds = [s.hold_ms for s in self.shots]
        rises = [s.rise_ms for s in self.shots]
        spread = f"  sd {pstdev(holds):.1f}/{pstdev(rises):.1f}" if len(self.shots) > 1 else ""
        return (f"{len(self.shots)} shots   mean hold {mean(holds):.1f} ms   "
                f"mean rise {mean(rises):.1f} ms{spread}")

    def report(self) -> None:
        if not self.shots:
            print("No shots detected.")
            return
        print()
        for shot in self.shots:
            print(shot.describe(self.expect_hold, self.expect_rise))
        print(self.summary())
        if self.expect_hold or self.expect_rise:
            print(f"expected: hold {self.expect_hold:.0f} ms (TOTAL_TIME_MP + latency_val), "
                  f"rise {self.expect_rise:.0f} ms (RISE_TIME_MP)")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
class Sampler(threading.Thread):
    """Polls the pad on its own thread so the plot can't slow down sampling."""

    def __init__(self, reader, window_ms: int, analyzer: ShotAnalyzer | None = None):
        super().__init__(daemon=True)
        maxlen = int(window_ms / 1000 * SAMPLE_HZ) + 1
        self.t = deque(maxlen=maxlen)
        self.rx = deque(maxlen=maxlen)
        self.ry = deque(maxlen=maxlen)
        self.log: list[tuple[float, float, float]] = []
        self.recording = False
        self.analyzer = analyzer
        self.new_shots: list[Shot] = []
        self._reader = reader
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self) -> None:
        period = 1.0 / SAMPLE_HZ
        t0 = time.perf_counter()
        while not self._stop.is_set():
            rx, ry = self._reader.read()
            t_ms = (time.perf_counter() - t0) * 1000.0
            shot = self.analyzer.feed(t_ms, rx, ry) if self.analyzer else None
            with self._lock:
                self.t.append(t_ms)
                self.rx.append(rx)
                self.ry.append(ry)
                if shot:
                    self.new_shots.append(shot)
                if self.recording:
                    self.log.append((t_ms, rx, ry))
            time.sleep(period)

    def snapshot(self):
        with self._lock:
            return list(self.t), list(self.rx), list(self.ry)

    def drain_shots(self) -> list[Shot]:
        with self._lock:
            shots, self.new_shots = self.new_shots, []
            return shots

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def setup_axes(window_ms: int):
    fig, (ax_t, ax_xy) = plt.subplots(1, 2, figsize=(12, 5))

    ax_t.set_title("Right stick over time")
    ax_t.set_xlabel("ms")
    ax_t.set_ylabel("value (-100..100)")
    ax_t.set_xlim(0, window_ms)
    ax_t.set_ylim(-110, 110)
    ax_t.grid(True)

    ax_xy.set_title("Stick position (physical orientation)")
    ax_xy.set_xlabel("RX")
    ax_xy.set_ylabel("RY (up = flick up)")
    ax_xy.set_xlim(-110, 110)
    ax_xy.set_ylim(-110, 110)
    ax_xy.set_aspect("equal")
    ax_xy.grid(True)
    ax_xy.add_patch(plt.Circle((0, 0), 100, fill=False, color="0.7"))
    return fig, ax_t, ax_xy


def live(args) -> None:
    reader = make_reader(args.backend, args.pad, args.axes, wait=not args.no_wait)
    print(f"Reading pad {reader.pad} via {reader.name}. Close the window to quit.")

    analyzer = ShotAnalyzer(args.expect_hold, args.expect_rise)
    sampler = Sampler(reader, args.window, analyzer)
    sampler.recording = args.record is not None
    sampler.start()

    fig, ax_t, ax_xy = setup_axes(args.window)
    (line_ry,) = ax_t.plot([], [], color="tab:blue", label="RY (+ = down)")
    (line_rx,) = ax_t.plot([], [], color="tab:orange", label="RX")
    ax_t.axhline(DOWN_LEVEL, color="0.8", lw=0.8)
    ax_t.axhline(UP_LEVEL, color="0.8", lw=0.8)
    ax_t.legend(loc="upper right")
    (trail,) = ax_xy.plot([], [], ".-", color="tab:blue", alpha=0.5, markersize=3)
    (dot,) = ax_xy.plot([], [], "o", color="tab:red", markersize=9)
    readout = ax_t.text(0.02, 0.03, "waiting for a shot...", transform=ax_t.transAxes,
                        family="monospace", fontsize=9, va="bottom",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.8"))

    def update(_frame):
        for shot in sampler.drain_shots():
            print(shot.describe(args.expect_hold, args.expect_rise))
            readout.set_text(
                f"last shot #{shot.index}\n"
                f"  hold {shot.hold_ms:7.1f} ms\n"
                f"  rise {shot.rise_ms:7.1f} ms\n"
                f"  up   {shot.up_hold_ms:7.1f} ms\n"
                f"{analyzer.summary()}"
            )
        t, rx, ry = sampler.snapshot()
        if not t:
            return line_rx, line_ry, trail, dot
        t_end = t[-1]
        t_start = max(0.0, t_end - args.window)
        rel = [v - t_start for v in t]
        line_rx.set_data(rel, rx)
        line_ry.set_data(rel, ry)
        ax_t.set_xlim(0, args.window)
        trail.set_data(rx, [-v for v in ry])
        dot.set_data([rx[-1]], [-ry[-1]])
        return line_rx, line_ry, trail, dot

    anim = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
    fig._rs_anim = anim  # keep a reference alive
    try:
        plt.show()
    finally:
        sampler.stop()
        if args.record:
            save_csv(args.record, sampler.log)
        analyzer.report()


def save_csv(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_ms", "rx", "ry"])
        writer.writerows((f"{t:.2f}", f"{rx:.2f}", f"{ry:.2f}") for t, rx, ry in rows)
    print(f"Saved {len(rows)} samples to {path}")


def replay(args) -> None:
    t, rx, ry = [], [], []
    with open(args.replay, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t_ms"]))
            rx.append(float(row["rx"]))
            ry.append(float(row["ry"]))
    if not t:
        raise SystemExit(f"{args.replay} has no samples.")

    analyzer = ShotAnalyzer(args.expect_hold, args.expect_rise)
    for sample in zip(t, rx, ry):
        analyzer.feed(*sample)
    analyzer.report()

    if args.no_plot:
        return

    fig, ax_t, ax_xy = setup_axes(int(t[-1]) or args.window)
    ax_t.set_xlim(t[0], t[-1])
    ax_t.plot(t, ry, color="tab:blue", label="RY (+ = down)")
    ax_t.plot(t, rx, color="tab:orange", label="RX")
    for shot in analyzer.shots:
        ax_t.axvspan(shot.t_start, shot.t_rise_start, color="tab:blue", alpha=0.12)
        ax_t.axvspan(shot.t_rise_start, shot.t_rise_end, color="tab:red", alpha=0.18)
        ax_t.annotate(f"{shot.hold_ms:.0f} / {shot.rise_ms:.0f}",
                      (shot.t_rise_start, 105), ha="center", fontsize=8)
    ax_t.set_title("Right stick over time (blue = hold down, red = rise)")
    ax_t.legend(loc="lower right")
    ax_xy.plot(rx, [-v for v in ry], ".-", color="tab:blue", alpha=0.6, markersize=3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend", choices=["auto", *BACKENDS], default="auto",
        help="auto tries xinput then winmm",
    )
    parser.add_argument("--pad", type=int, help="controller index (default: first connected)")
    parser.add_argument(
        "--axes",
        default=None,
        help="RX,RY axes - letters for winmm (default u,r), indices for pygame (default 2,3)",
    )
    parser.add_argument("--window", type=int, default=2000, help="time window in ms")
    parser.add_argument("--record", metavar="CSV", help="save samples to a CSV file")
    parser.add_argument("--replay", metavar="CSV", help="plot a previously recorded CSV")
    parser.add_argument("--list", action="store_true", help="show detected controllers and exit")
    parser.add_argument("--monitor", action="store_true", help="print raw axis values instead of plotting")
    parser.add_argument("--no-wait", action="store_true", help="fail instead of waiting for a pad")
    parser.add_argument("--no-plot", action="store_true", help="with --replay, print timings only")
    parser.add_argument(
        "--expect-hold", type=float, default=EXPECT_HOLD_MS,
        help=f"expected hold-down ms for the delta column (v5: {EXPECT_HOLD_MS:.0f}, 0 = off)",
    )
    parser.add_argument(
        "--expect-rise", type=float, default=EXPECT_RISE_MS,
        help=f"expected rise ms for the delta column (v5: {EXPECT_RISE_MS:.0f}, 0 = off)",
    )
    args = parser.parse_args()

    default_axes = "2,3" if args.backend == "pygame" else "u,r"
    args.axes = tuple((args.axes or default_axes).split(","))

    if args.list:
        list_devices()
    elif args.replay:
        replay(args)
    elif args.monitor:
        monitor(make_reader(args.backend, args.pad, args.axes, wait=not args.no_wait))
    else:
        live(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
