"""Lab setup, I/O, timing, logging and plotting. Controllers live in the notebook.

q: rad, v: rad/s, effort: duty fraction (XL430 command) or A (XL330).
XL430 measured effort is *load fraction*, not PWM or calibrated torque.
No port is opened and no torque is enabled on import.
"""
import asyncio
from collections import deque
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
import csv
import math
import time
import warnings


def clip(value, limit):
    if not math.isfinite(value) or not math.isfinite(limit) or limit < 0:
        raise ValueError("finite value and nonnegative finite limit required")
    return max(-limit, min(limit, value))


def wrap_angle(angle):
    """Radians in [-pi, pi); also used to unwrap successive encoder samples."""
    if not math.isfinite(angle):
        raise ValueError("finite angle required")
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class State:
    q: float
    v: float
    effort: float
    pwm: float = 0.0  # signed actual duty fraction, Present PWM / 885


@dataclass(frozen=True)
class Station:
    name: str
    family: str
    model: int
    kp: float
    kd: float
    effort_limit: float
    pwm_limit: float = 0.8
    period: float = 0.01
    travel: float = math.radians(75)

    @property
    def command_unit(self):
        return "duty fraction" if self.family == "A" else "A"

    @property
    def measured_unit(self):
        return "estimated load fraction" if self.family == "A" else "A (supply current)"


# Starting values only: these require instructor commissioning on the actual rigs.
STATIONS = {
    name: Station(name, family, 1060 if family == "A" else 1200,
                  0.12 if family == "A" else 0.08,
                  0.008 if family == "A" else 0.006,
                  0.12 if family == "A" else 0.08)
    for name, family in (("A1", "A"), ("A2", "A"), ("B1", "B"),
                         ("B2", "B"), ("S", "S"))
}


@contextmanager
def stop_on_interrupt():
    """Let a notebook Stop/Interrupt end one trial and continue to its plots.

    Use outside running(): torque is disabled before the interrupt is handled.
    Other exceptions still propagate.
    """
    try:
        yield
    except KeyboardInterrupt:
        print("Stopped by operator. Torque off; plotting the recorded motion.")


class Rig:
    """One homogeneous pair, or the single XL330 leader at S.

    Place both horns at the desired neutral before open(). Logical coordinates
    are relative to those encoders; directions maps mirrored mountings to +q.
    Modes and EEPROM limits are set only when explicitly configuring a run.
    """

    def __init__(self, port, station="B1", ids=(1, 2), baud=1_000_000,
                 directions=None, bus=None):
        self.station = STATIONS[station]
        self.ids = tuple(ids)
        expected = 1 if self.station.family == "S" else 2
        if len(self.ids) != expected or len(set(self.ids)) != expected:
            raise ValueError(f"{station} needs {expected} distinct motor IDs")
        self.directions = dict(directions or {i: 1 for i in self.ids})
        if set(self.directions) != set(self.ids) or any(v not in (-1, 1) for v in self.directions.values()):
            raise ValueError("provide direction +1 or -1 for each ID")
        if bus is None:
            from dxl_bus import DynamixelBus
            bus = DynamixelBus(port, baud)
        self.bus = bus
        self.zeros = {}
        self._last_encoder = {}
        self._position = {}
        self.modes = {}
        self.active = False
        self.enabled = {i: False for i in self.ids}
        self.watchdog_ms = 200

    def open(self):
        self.bus.open()
        try:
            for i in self.ids:
                model = self.bus.ping(i)
                self.bus.torque(i, False)
                if model != self.station.model:
                    raise ValueError(f"ID {i}: model {model}; {self.station.name} expects {self.station.model}")
                if self.bus.hardware_error(i):
                    raise IOError(f"ID {i}: hardware error; inspect power/temperature/load")
                self.bus.clear_bus_watchdog(i)
                # Establish a stable single-turn encoder frame, also for current/PWM.
                from dxl_bus import MODE
                self.bus.set_mode(i, MODE["POSITION"])
            self.zero_here()
            return self
        except BaseException:
            self.close()
            raise

    def encoder_status(self):
        """Raw and continuous relative encoder angles, in degrees.

        This only reads encoders: it never commands motion or changes the zero.
        encoder_deg uses the low-level convention (2048 ticks = 0 degrees).
        """
        raw = self.bus.read_states(self.ids)
        positions = self._track_positions(raw)
        return {
            i: dict(encoder_deg=math.degrees(raw[i][0]),
                    zero_deg=math.degrees(self.zeros[i]) if i in self.zeros else None,
                    relative_deg=math.degrees(positions[i]),
                    position_command_limit_deg=math.degrees(self.station.travel))
            for i in self.ids
        }

    def zero_here(self):
        """Torque off, then explicitly make the current horn positions neutral.

        Place both horns at their intended physical neutral before calling.
        All zeros are validated before replacing any. Reconfigure the next trial
        afterwards; old position goals must not survive a coordinate change.
        """
        self.stop()
        # Verify torque-off succeeded: stop() otherwise warns on failed writes.
        from dxl_bus import ADDR
        for i in self.ids:
            if self.bus.read(i, ADDR["TORQUE_ENABLE"], 1, signed=False):
                raise RuntimeError(f"ID {i}: cannot re-zero while torque is enabled")
        zeros = {i: x[0] for i, x in self.bus.read_states(self.ids).items()}
        for i, zero in zeros.items():
            if not math.isfinite(zero):
                raise ValueError(f"ID {i}: nonfinite encoder; no new zero was stored")
        self.zeros = zeros
        self._last_encoder = dict(zeros)
        self._position = {i: 0. for i in self.ids}
        self.modes = {}
        return self.encoder_status()

    def _track_positions(self, raw):
        # Track BOTH motors independently. A crossing from +179 to -179 is
        # +2 degrees, not -358. Continuous SDK multi-turn values work too.
        # Unwrapping assumes less than half a revolution between samples.
        for i in self.ids:
            if not all(math.isfinite(x) for x in raw[i]):
                raise RuntimeError(f"ID {i}: nonfinite encoder state")
        positions = {
            i: self._position[i] + self.directions[i]*wrap_angle(raw[i][0]-self._last_encoder[i])
            for i in self.ids
        }
        self._position = positions
        self._last_encoder = {i: raw[i][0] for i in self.ids}
        return positions

    def read(self):
        """Continuous relative angles on both motors, without a +/-75 read cutoff."""
        raw = self.bus.read_states_with_pwm(self.ids)
        positions = self._track_positions(raw)
        states = {}
        for i in self.ids:
            _, v, effort, pwm = raw[i]
            sign = self.directions[i]
            # Both models have signed effort at 126, with different scale factors.
            states[i] = State(positions[i], sign * v, sign * effort * 0.001, sign * pwm/885)
        return states

    def configure(self, modes, position_kp=150):
        """Set per-ID 'off', 'effort' or 'position', without enabling torque.

        Position mode uses low firmware Kp; effort mode uses host PWM/current.
        Limits are EEPROM writes only if the installed value differs.
        """
        from dxl_bus import MODE, ADDR
        if set(modes) != set(self.ids) or any(m not in ("off", "effort", "position") for m in modes.values()):
            raise ValueError("specify off/effort/position for every motor")
        self.stop()
        try:
            for i, mode in modes.items():
                self.bus.clear_bus_watchdog(i)
                raw_mode = MODE["POSITION"] if mode in ("off", "position") else (
                    MODE["PWM"] if self.station.family == "A" else MODE["CURRENT"])
                # Avoid repeatedly resetting the encoder of an already torque-off leader.
                if self.bus.read(i, ADDR["OPERATING_MODE"], 1, signed=False) != raw_mode:
                    self.bus.set_mode(i, raw_mode)
                # Firmware can automatically enable torque when a goal is written.
                # Disable only that Drive Mode bit, preserving direction/profile bits.
                drive_mode = self.bus.read(i, ADDR["DRIVE_MODE"], 1, signed=False)
                if drive_mode & 0x08:
                    self.bus.write(i, ADDR["DRIVE_MODE"], 1, drive_mode & ~0x08)
                    if self.bus.read(i, ADDR["DRIVE_MODE"], 1, signed=False) & 0x08:
                        raise IOError(f"ID {i}: could not disable Torque On by Goal Update")
                pwm_raw = round(self.station.pwm_limit * 885)
                if self.bus.read(i, ADDR["PWM_LIMIT"], 2, signed=False) != pwm_raw:
                    self.bus.set_pwm_limit(i, self.station.pwm_limit)
                if self.station.family != "A":
                    ma = round(1000 * self.station.effort_limit)
                    if self.bus.read(i, ADDR["CURRENT_LIMIT"], 2, signed=False) != ma:
                        self.bus.set_current_limit(i, ma)
                if mode == "position":
                    self.bus.set_position_gains(i, kp=position_kp, kd=0, ki=0)
                    # In Position mode Goal PWM is an output ceiling, not effort.
                    self.bus.command_pwm(i, self.station.pwm_limit)
                    self.bus.no_profile(i)
                    self.bus.command_position(i, self.bus.read_position(i))
                elif mode == "effort":
                    if self.station.family == "A":
                        self.bus.command_pwm(i, 0)
                    else:
                        self.bus.command_current(i, 0)
                self.bus.set_bus_watchdog(i, self.watchdog_ms)
            self.modes = dict(modes)
        except BaseException:
            self.stop()
            raise

    @contextmanager
    def running(self, torque_off=()):
        """Enable configured motors only for the duration of a visible loop."""
        if not self.modes:
            raise RuntimeError("configure the rig before running")
        try:
            for i, mode in self.modes.items():
                self.bus.clear_bus_watchdog(i)
                self.bus.set_bus_watchdog(i, self.watchdog_ms)
                if mode != "off" and i not in torque_off:
                    self.bus.torque(i, True)
                    self.enabled[i] = True
            self.active = True
            yield self
        finally:
            self.stop()

    def enable_effort(self, motor, enabled):
        """Switch one configured effort motor; write its goal before enabling."""
        if not self.active or self.modes.get(motor) != "effort":
            raise RuntimeError("torque switching requires a running effort motor")
        if self.enabled[motor] != bool(enabled):
            self.bus.torque(motor, bool(enabled))
            from dxl_bus import ADDR
            actual = self.bus.read(motor, ADDR["TORQUE_ENABLE"], 1, signed=False)
            if bool(actual) != bool(enabled):
                raise IOError(f"ID {motor}: torque-enable readback did not match request")
            self.enabled[motor] = bool(enabled)

    def command(self, effort=None, position=None):
        """One sync write per command type. Returns applied, clipped commands."""
        effort, position = dict(effort or {}), dict(position or {})
        for values, mode in ((effort, "effort"), (position, "position")):
            if any(self.modes.get(i) != mode for i in values):
                raise ValueError(f"command does not match configured {mode} IDs")
        if not self.active:
            raise RuntimeError("commands require a running() context")
        effort = {i: clip(v, self.station.effort_limit) for i, v in effort.items()}
        position = {i: clip(v, self.station.travel) for i, v in position.items()}
        # Firmware Position mode is single-turn. Bound its raw goal rather than
        # wrapping a goal across the seam and requesting an almost-full turn.
        raw_position = {
            i: max(-math.pi, min(math.pi-2*math.pi/4096,
                                wrap_angle(self.zeros[i]) + self.directions[i]*v))
            for i, v in position.items()
        }
        position = {i: self.directions[i]*(v-wrap_angle(self.zeros[i]))
                    for i, v in raw_position.items()}
        # Validate all values before the first write.
        if effort:
            raw = {i: self.directions[i] * v for i, v in effort.items()}
            if self.station.family == "A":
                self.bus.sync_pwm(raw)
            else:
                self.bus.sync_current({i: 1000 * v for i, v in raw.items()})
        if position:
            self.bus.sync_position(raw_position)
        return effort | position

    def stop(self):
        self.active = False
        errors = []
        for i in self.ids:
            try:
                self.bus.torque(i, False)
                self.enabled[i] = False
            except Exception as exc:
                errors.append(f"ID {i}: {exc}")
        if errors:
            warnings.warn("Torque-off failed: " + "; ".join(errors) + ". Cut motor power.", RuntimeWarning)

    def close(self):
        try:
            self.stop()
        finally:
            self.bus.port.closePort()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class LoopTimer:
    """Monotonic scheduler; reports measured dt and skips missed deadlines.

    realtime=False advances nominal time for a deterministic software demo.
    Python/USB is not a hard real-time controller. Inspect stats after each run.
    """
    def __init__(self, period=0.01, realtime=True):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period must be positive")
        self.period, self.realtime = period, realtime
        self.start = self.last = time.perf_counter()
        self.deadline = self.start + period
        self.t, self.dt, self.samples, self.overruns = 0., period, 0, 0
        self.intervals = []
        self.work_times = []
        self.over_budget = 0

    def wait(self):
        if self.realtime:
            now = time.perf_counter()
            work = now-self.last
            self.work_times.append(work)
            self.over_budget += int(work > self.period)
            if now < self.deadline:
                time.sleep(self.deadline - now)
            now = time.perf_counter()
            self.dt = now - self.last
            self.t = now - self.start
            self.last = now
            self.deadline += self.period
            if now >= self.deadline:
                missed = int((now - self.deadline) / self.period) + 1
                self.overruns += missed
                self.deadline += missed * self.period
        else:
            self.dt = self.period
            self.t = (self.samples + 1) * self.period
        self.samples += 1
        self.intervals.append(self.dt)
        if self.realtime and self.dt > 0.15:
            raise RuntimeError("Loop stalled >150 ms; stopping before the next command")

    def stats(self):
        ds = sorted(self.intervals)
        mean = sum(ds)/len(ds) if ds else 0.
        return {"timebase": "wall_clock" if self.realtime else "synthetic",
                "requested_hz": 1/self.period,
                "achieved_hz": 1/mean if self.realtime and mean else None,
                "samples": self.samples, "missed_deadlines": self.overruns,
                "over_budget_cycles": self.over_budget,
                "mean_dt_ms": 1000*mean,
                "p95_dt_ms": 1000*ds[min(len(ds)-1, int(.95*len(ds)))] if ds else 0,
                "p99_dt_ms": 1000*ds[min(len(ds)-1, int(.99*len(ds)))] if ds else 0,
                "max_dt_ms": 1000*ds[-1] if ds else 0,
                "jitter_std_ms": 1000*math.sqrt(sum((d-mean)**2 for d in ds)/len(ds)) if ds else 0,
                "mean_work_ms": 1000*sum(self.work_times)/len(self.work_times) if self.work_times else 0}

    def report(self):
        s = self.stats()
        if not self.realtime:
            return f"Preview: synthetic {self.period*1000:g} ms steps; hardware rate is not measured."
        if s["achieved_hz"] is None:
            return "No complete control periods measured."
        return (f"Requested {s['requested_hz']:.0f} Hz ({1000*self.period:g} ms); "
                f"achieved {s['achieved_hz']:.0f} Hz. "
                f"Period mean/p99/max: {s['mean_dt_ms']:.3f}/{s['p99_dt_ms']:.3f}/{s['max_dt_ms']:.3f} ms; "
                f"jitter SD: {s['jitter_std_ms']:.3f} ms; "
                f"skipped periods: {self.overruns}; work over budget: {self.over_budget} cycles.")


class Log:
    def __init__(self):
        self.rows = []

    def add(self, **row):
        if self.rows and row.keys() != self.rows[0].keys():
            raise ValueError("all log rows must have the same fields")
        self.rows.append(row)

    def save(self, path):
        if not self.rows:
            raise ValueError("empty log")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def metrics(self):
        if not self.rows:
            return {}
        if "xF" in self.rows[0]:
            errors = [r["x_des"]-r["xF"] for r in self.rows]
            return {"tracking_rmse_m": math.sqrt(sum(e*e for e in errors)/len(errors)),
                    "max_error_m": max(map(abs, errors))}
        errors = [r.get("q_des", r["qL"]) - r["qF"] for r in self.rows]
        return {"tracking_rmse_rad": math.sqrt(sum(e*e for e in errors)/len(errors)),
                "max_error_rad": max(map(abs, errors))}


def plot(log, title="Teleoperation", effort_unit="command units", measured_unit=None):
    """Positions, desired tracking error, commands, and optional measurements."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if not log.rows:
        raise ValueError("empty log")
    rows, fields = log.rows, log.rows[0]
    measured = measured_unit is not None and "mL" in fields
    has_pwm = "pwmL" in fields
    timing = "dt_s" in fields
    panels = 3 + int(measured) + int(has_pwm) + int(timing)
    fig = make_subplots(rows=panels, cols=1, shared_xaxes=True,
                        subplot_titles=["Position", "Tracking error", "Applied command"] +
                        (["Measured effort (not calibrated torque)"] if measured else []) +
                        (["Present PWM: actual output, not torque"] if has_pwm else []) +
                        (["Measured loop period" if fields.get("wall_clock") else "Synthetic loop period"] if timing else []))
    t = [r["t"] for r in rows]
    for key in ("qL", "qF", "q_des", "leader_target"):
        if key in fields:
            if key == "leader_target" and not any(abs(r[key]-r["qL"]) > 1e-9 for r in rows):
                continue
            fig.add_trace(go.Scatter(x=t, y=[r[key] for r in rows], name=key), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[r.get("q_des", r["qL"])-r["qF"] for r in rows], name="error"), row=2, col=1)
    for key in ("uL", "uF"):
        if key in fields:
            fig.add_trace(go.Scatter(x=t, y=[r[key] for r in rows], name=key), row=3, col=1)
    if measured:
        for key in ("mL", "mF", "reaction"):
            if key not in fields:
                continue
            fig.add_trace(go.Scatter(x=t, y=[r[key] for r in rows], name=key), row=4, col=1)
        fig.update_yaxes(title_text=measured_unit, row=4, col=1)
    if has_pwm:
        pwm_row = 5 if measured else 4
        for key in ("pwmL", "pwmF"):
            fig.add_trace(go.Scatter(x=t, y=[r[key] for r in rows], name=key), row=pwm_row, col=1)
        fig.update_yaxes(title_text="PWM duty fraction", row=pwm_row, col=1)
    if timing:
        fig.add_trace(go.Scatter(x=t, y=[1000*r["dt_s"] for r in rows], name="loop period"), row=panels, col=1)
        fig.add_hline(y=1000*fields["requested_T"], line_dash="dash", row=panels, col=1)
        fig.update_yaxes(title_text="ms", row=panels, col=1)
    for r in (1, 2):
        fig.update_yaxes(title_text="rad", row=r, col=1)
    fig.update_yaxes(title_text=effort_unit, row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=panels, col=1)
    fig.update_layout(title=title, height=260*panels)
    return fig


def plot_insertion(log, title="Robot insertion", coordinate_name="Instrument insertion"):
    """Keep rotational leader, linear follower and contact units separate."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = log.rows
    t = [r["t"] for r in rows]
    timing = "dt_s" in rows[0]
    panels = 5 if timing else 4
    fig = make_subplots(rows=panels, cols=1, shared_xaxes=True,
                        specs=[[{}], [{}], [{"secondary_y": True}], [{}]] + ([[{}]] if timing else []),
                        subplot_titles=[coordinate_name, "Contact and commanded axial force",
                                        "Leader current feedback", "Leader angle"] +
                        (["Measured loop period" if rows[0]["wall_clock"] else "Synthetic loop period"] if timing else []))
    for key in ("x_des", "xF"):
        fig.add_trace(go.Scatter(x=t, y=[1000*r[key] for r in rows], name=key), row=1, col=1)
    for key in ("force_N", "uF"):
        fig.add_trace(go.Scatter(x=t, y=[r[key] for r in rows], name=key), row=2, col=1)
    if "virtual_force_N" in rows[0] and any(abs(r["virtual_force_N"]) > 1e-9 for r in rows):
        fig.add_trace(go.Scatter(x=t, y=[r["virtual_force_N"] for r in rows],
                                 name="rendered wall law"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=[r["uL"] for r in rows], name="leader current"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=[r["mL"] for r in rows], name="measured supply current"), row=3, col=1)
    if "i_ff" in rows[0]:
        fig.add_trace(go.Scatter(x=t, y=[r["i_ff"] for r in rows],
                                 name="requested friction assistance", line_dash="dash"), row=3, col=1)
    if "leader_enabled" in rows[0]:
        fig.add_trace(go.Scatter(x=t, y=[r["leader_enabled"] for r in rows],
                                 name="leader torque enabled", line_shape="hv"),
                      row=3, col=1, secondary_y=True)
        fig.update_yaxes(title_text="motor", tickvals=[0, 1], ticktext=["off", "on"],
                         range=[-.05, 1.05], row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=[math.degrees(r["qL"]) for r in rows], name="leader angle"), row=4, col=1)
    for index, unit in enumerate(("mm", "N", "A", "degrees"), 1):
        fig.update_yaxes(title_text=unit, row=index, col=1)
    if timing:
        fig.add_trace(go.Scatter(x=t, y=[1000*r["dt_s"] for r in rows], name="loop period"), row=5, col=1)
        fig.add_hline(y=1000*rows[0]["requested_T"], line_dash="dash", row=5, col=1)
        fig.update_yaxes(title_text="ms", row=5, col=1)
    fig.update_xaxes(title_text="Time (s)", row=panels, col=1)
    fig.update_layout(title=title, height=250*panels)
    return fig


class Clutch:
    """S ends a run normally; held Space optionally acts as a clutch.

    In headless sessions, use a timed clutch in the notebook. No silent fallback.
    """
    def __init__(self):
        self.held = self.stop_requested = False
        self.listener = None

    def __enter__(self):
        from pynput import keyboard
        def press(key):
            if key == keyboard.Key.space:
                self.held = True
            elif getattr(key, "char", "") in ("s", "S"):
                self.stop_requested = True
        def release(key):
            if key == keyboard.Key.space:
                self.held = False
        self.listener = keyboard.Listener(on_press=press, on_release=release)
        self.listener.start()
        self.listener.wait()
        return self

    def __exit__(self, *exc):
        if self.listener:
            self.listener.stop()
            self.listener.join(timeout=1)


class NotebookStop:
    """Widget callback only requests exit; the loop owns motor I/O and cleanup."""
    def __init__(self, trial):
        self.trial = trial

    def __enter__(self):
        from ipywidgets import Button
        from IPython.display import display
        self.button = Button(description="Stop", button_style="danger", icon="stop")
        self.button.on_click(self._clicked)
        display(self.button)
        return self

    def _clicked(self, button):
        self.trial.request_stop("Stop button")

    def __exit__(self, *exc):
        self.button.on_click(self._clicked, remove=True)
        self.button.disabled = True
        self.button.description = "Stopped"


class Delay:
    """N-sample delay on one or more channels; startup holds the first sample.

    Usage: delay = Delay(samples=3); remoteL, remoteF = delay(L, F).
    Local states remain undelayed in the student's control equations.
    """
    def __init__(self, samples=0):
        if not isinstance(samples, int) or samples < 0:
            raise ValueError("samples must be a nonnegative integer")
        self.samples = samples
        self._queue = None

    def __call__(self, *states):
        if self._queue is None:
            self._queue = deque([states]*(self.samples+1), maxlen=self.samples+1)
        elif len(states) != len(self._queue[0]):
            raise ValueError("keep the same number of delayed channels")
        self._queue.append(states)
        return self._queue[0]


class ContactGate:
    """Provided lab scaffold: engage in compression and release on withdrawal.

    The instructor notebook exposes this state logic. Student cells only supply
    the force-feedback equation; the hardware switching stays in the helper.
    """
    def __init__(self, engage=.3, release=.1, velocity_deadband=.02):
        if not (math.isfinite(engage) and math.isfinite(release)
                and math.isfinite(velocity_deadband)
                and engage > release >= 0 and velocity_deadband >= 0):
            raise ValueError("require engage > release >= 0 and nonnegative velocity deadband")
        self.engage, self.release, self.velocity_deadband = engage, release, velocity_deadband
        self.advancing = self.active = False

    def __call__(self, velocity, force):
        if not math.isfinite(velocity) or not math.isfinite(force):
            raise ValueError("finite velocity and contact force required")
        if abs(velocity) > self.velocity_deadband:
            self.advancing = velocity > 0
        compression = max(0., -force)
        self.active = self.advancing and compression > (self.release if self.active else self.engage)
        return self.active


class ForceGate:
    """Contact-only torque with force hysteresis, independent of motion direction."""
    def __init__(self, engage=.1, release=.02):
        if not (math.isfinite(engage) and math.isfinite(release) and engage > release >= 0):
            raise ValueError("require engage > release >= 0")
        self.engage, self.release, self.active = engage, release, False

    def __call__(self, force):
        if not math.isfinite(force):
            raise ValueError("finite contact force required")
        self.active = max(0., -force) > (self.release if self.active else self.engage)
        return self.active


@dataclass(frozen=True)
class JointState(State):
    # Environment-on-joint reaction in command units. At B it is -current;
    # at S it is the simulated contact torque times the chosen reflection gain.
    reaction: float = 0.0
    x: float = 0.0       # simulated insertion [m]
    xdot: float = 0.0    # insertion velocity [m/s]
    force: float = 0.0   # environment-on-tool axial force [N]


class Lab:
    """Notebook plumbing only. No teleoperation controller is implemented here.

    Each run handles modes, optional keyboard input, timing, logging and plots.
    read()/write()/wait() remain explicit in the student's while loop.
    """
    def __init__(self, port, station="B1", baud=1_000_000, leader=1, follower=2,
                 directions=None, demo=False, keyboard=True, rig=None,
                 show_plots=True, sim_gui=None, tissue="bone", metres_per_rad=.04,
                 force_feedback_a_per_n=.01, scene="insertion", current_limit=None, pwm_limit=None, stop_button=False):
        self.cfg = STATIONS[station]
        if current_limit is not None:
            if self.cfg.family == "A" or not math.isfinite(current_limit) or not 0 < current_limit <= 1.75:
                raise ValueError("current_limit must be XL330 current in A, within (0, 1.75]")
            self.cfg = replace(self.cfg, effort_limit=current_limit)
        if pwm_limit is not None:
            if not math.isfinite(pwm_limit) or not 0 < pwm_limit <= 1:
                raise ValueError("pwm_limit must be a duty fraction within (0, 1]")
            self.cfg = replace(self.cfg, pwm_limit=pwm_limit,
                               effort_limit=pwm_limit if self.cfg.family == "A" else self.cfg.effort_limit)
        self.leader_id, self.follower_id = leader, follower
        self.demo, self.keyboard, self.show_plots = demo, keyboard, show_plots
        self.stop_button = stop_button
        self.sim_gui = (not demo) if sim_gui is None else sim_gui
        self.realtime = not demo or (self.cfg.family == "S" and self.sim_gui)
        self.metres_per_rad = metres_per_rad
        self.force_feedback_a_per_n = force_feedback_a_per_n
        if scene not in ("insertion", "sphere-wall"):
            raise ValueError("scene must be insertion or sphere-wall")
        self.scene = scene
        self.results, self.settings = {}, {}
        self.sim, self._trial = None, None
        ids = (leader,) if self.cfg.family == "S" else (leader, follower)
        if rig is None:
            if demo:
                from demo_rig import DemoRig
                rig = DemoRig(station, ids)
            else:
                rig = Rig(port, station, ids, baud, directions=directions)
        if current_limit is not None or pwm_limit is not None:
            rig.station = self.cfg
        self.rig = rig
        try:
            rig.open()
            if self.cfg.family == "S":
                if scene == "sphere-wall":
                    from sim_wall import SphereWall
                    self.sim = SphereWall(self.cfg.period, gui=self.sim_gui)
                else:
                    from sim_insertion import InsertionFollower
                    self.sim = InsertionFollower(self.cfg.period, tissue=tissue, gui=self.sim_gui,
                                                 metres_per_rad=metres_per_rad)
        except BaseException:
            self.close()
            raise
        units = f"leader A; {scene} follower N and m" if self.sim is not None else self.cfg.command_unit
        print(f"{station}: {'software preview' if demo else port}; torque off. "
              f"Gains/output use {units}. Run one experiment cell.")

    def run(self, name, leader="off", follower="effort", dt=.01,
            duration=None, clutch=False, only=None, position_kp=150, tissue=None, wall_stiffness=None):
        """Prepare one trial. None duration = until Stop (10 s in demo mode)."""
        return Trial(self, name, leader, follower, dt, duration, clutch, only, position_kp, tissue, wall_stiffness)

    def figure(self, log, name):
        if self.sim is not None:
            return plot_insertion(log, name, "Sphere position" if self.scene == "sphere-wall" else "Instrument insertion")
        return plot(log, name, self.cfg.command_unit, self.cfg.measured_unit)

    def zero_here(self):
        return self.rig.zero_here()

    def close(self):
        try:
            if self._trial is not None:
                self._trial._stack.close()
                self._trial = None
            self.rig.close()
        finally:
            if self.sim is not None:
                self.sim.close()

    def export(self, directory="lab_results"):
        """Save accumulated trials as CSV + standalone HTML + run metadata."""
        from pathlib import Path
        import json
        folder = Path(directory) / (self.cfg.name + ("_demo" if self.demo else "_hardware"))
        folder.mkdir(parents=True, exist_ok=True)
        for name, log in self.results.items():
            if not log.rows:
                continue
            log.save(folder / f"{name}.csv")
            self.figure(log, name).write_html(
                folder / f"{name}.html", include_plotlyjs=True)
        (folder / "settings.json").write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        print("Saved", folder.resolve())
        return folder


class Trial:
    """I/O and lifecycle for one visible notebook loop; contains no control law."""
    def __init__(self, lab, name, leader, follower, dt, duration, clutch, only, position_kp=150, tissue=None, wall_stiffness=None):
        self.lab, self.name = lab, name
        self.leader_mode, self.follower_mode = leader, follower
        self.period = dt
        self.position_kp = position_kp
        self.tissue = tissue
        self.wall_stiffness = wall_stiffness
        self.duration = (10. if lab.demo else None) if duration is None else duration
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        if self.duration is not None and (not math.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("duration must be positive or None")
        self.use_clutch = clutch
        self.enabled = only is None or lab.cfg.family in only
        self.log = Log()
        self._stack = ExitStack()
        self._key = None
        self._button = None
        self._stop_requested = False
        self.stop_reason = None
        self._next_key_poll = 0.
        self._gui_space = False
        self.leader = self.follower = JointState(0., 0., 0.)
        self.released = self._was_clutched = False

    def __enter__(self):
        lab = self.lab
        if not self.enabled:
            print(f"Skip {self.name}: this experiment belongs to another station.")
            return self
        if lab._trial is not None:
            raise RuntimeError("stop the current trial first")
        try:
            modes = {lab.leader_id: "effort" if self.leader_mode == "contact" else self.leader_mode}
            if lab.sim is None:
                modes[lab.follower_id] = self.follower_mode
            elif self.follower_mode != "effort":
                raise ValueError("simulated follower uses effort commands")
            lab.rig.configure(modes, position_kp=self.position_kp)
            if lab.sim is not None:
                lab.sim.set_period(self.period)
                if self.wall_stiffness is not None:
                    if lab.scene != "sphere-wall":
                        raise ValueError("wall_stiffness applies only to the sphere-wall scene")
                    lab.sim.set_stiffness(self.wall_stiffness)
                if self.tissue is not None:
                    lab.sim.set_tissue(self.tissue)
                else:
                    lab.sim.reset()
            gui_input = lab.sim is not None and lab.sim.gui
            # GUI simulations capture S in the existing PyBullet window.
            # Pair notebooks use an in-cell button, serviced by wait_async().
            if lab.stop_button:
                self._button = self._stack.enter_context(NotebookStop(self))
            if not lab.demo and lab.keyboard and (self.use_clutch or (not gui_input and not lab.stop_button)):
                try:
                    self._key = self._stack.enter_context(Clutch())
                except Exception as exc:
                    raise RuntimeError("S-key input needs desktop keyboard access") from exc
            if self.use_clutch and not lab.demo and self._key is None and not gui_input:
                raise RuntimeError("held-key clutch needs desktop keyboard input")
            self.leader, self.follower = self.read()
            torque_off = (lab.rig.ids if self._stop_requested else
                          ((lab.leader_id,) if self.leader_mode == "contact" else ()))
            self._stack.enter_context(lab.rig.running(torque_off=torque_off))
            self.timer = LoopTimer(self.period, realtime=lab.realtime)
            lab._trial = self
            stop_hint = "Focus PyBullet and press S" if gui_input else "Press S"
            if self._button is not None:
                stop_hint = "Click the Stop button above"
            elif not gui_input and self._key is None:
                stop_hint = "Preview ends automatically" if lab.demo else "Use notebook Stop (keyboard disabled)"
            print(f"{self.name}: target {1/self.period:g} Hz (T={1000*self.period:g} ms). "
                  f"{stop_hint}; torque off and plots follow.")
            return self
        except BaseException:
            self._stack.close()
            lab.rig.stop()
            raise

    @property
    def active(self):
        if not self.enabled:
            return False
        if self._key is not None and self._key.stop_requested:
            self.request_stop("S")
        sim = self.lab.sim
        if sim is not None and sim.gui and not self._stop_requested:
            now = time.perf_counter()
            if now >= self._next_key_poll:
                self._next_key_poll = now + .01
                if not sim.p.isConnected(sim.client):
                    self.request_stop("window closed")
                else:
                    try:
                        keys = sim.p.getKeyboardEvents(physicsClientId=sim.client)
                        pressed = sim.p.KEY_WAS_TRIGGERED | sim.p.KEY_IS_DOWN
                        if (keys.get(ord("s"), 0) | keys.get(ord("S"), 0)) & pressed:
                            self.request_stop("S")
                        if keys.get(32, 0) & pressed:
                            self._gui_space = True
                        if keys.get(32, 0) & sim.p.KEY_WAS_RELEASED:
                            self._gui_space = False
                    except sim.p.error:
                        self.request_stop("window closed")
        return (self.enabled and (self.duration is None or self.timer.t < self.duration)
                and not self._stop_requested)

    def request_stop(self, reason="requested"):
        """Only set a flag; motor I/O and plots stay on the control-loop thread."""
        self._stop_requested = True
        self.stop_reason = reason

    @property
    def dt(self):
        return self.timer.dt

    @property
    def clutched(self):
        if not self.use_clutch:
            return False
        return 3. <= self.timer.t < 5. if self.lab.demo else (self._key.held if self._key else self._gui_space)

    def read(self):
        lab = self.lab
        states = lab.rig.read()
        L = states[lab.leader_id]
        self.leader = JointState(L.q, L.v, L.effort, pwm=L.pwm, reaction=-L.effort)
        if lab.sim is None:
            F = states[lab.follower_id]
            self.follower = JointState(F.q, F.v, F.effort, pwm=F.pwm, reaction=-F.effort)
        else:
            F = lab.sim.read()
            self.follower = JointState(F.x/lab.metres_per_rad, F.velocity/lab.metres_per_rad,
                                       F.applied_force, reaction=F.force*lab.force_feedback_a_per_n,
                                       x=F.x, xdot=F.velocity, force=F.force)
        if hasattr(self, "timer"):
            held = self.clutched
            self.released = self._was_clutched and not held
            self._was_clutched = held
        return self.leader, self.follower

    def write(self, leader=None, follower=None, leader_position=None,
              follower_position=None, target=None, feedforward=0.0, leader_enabled=None, virtual_force=0.):
        lab = self.lab
        if self.leader_mode == "contact" and leader_enabled is None:
            raise ValueError("contact mode needs an explicit leader_enabled decision in the loop")
        if leader_enabled and leader is None:
            raise ValueError("write an explicit leader goal before enabling torque")
        if leader_enabled is not None:
            if self.leader_mode not in ("effort", "contact"):
                raise ValueError("leader_enabled applies only to current/PWM modes")
            if not leader_enabled:
                lab.rig.enable_effort(lab.leader_id, False)
                leader = 0.0
        efforts = {lab.leader_id: leader} if leader is not None else {}
        positions = {lab.leader_id: leader_position} if leader_position is not None else {}
        if lab.sim is None:
            if follower is not None:
                efforts[lab.follower_id] = follower
            if follower_position is not None:
                positions[lab.follower_id] = follower_position
        elif follower is None:
            raise ValueError("send simulated follower effort once per loop")
        # Rig validates all physical commands and clips before transmission.
        if lab.sim is not None:
            follower = clip(follower, lab.sim.force_limit)
        applied = lab.rig.command(effort=efforts, position=positions)
        if leader_enabled:
            # The clipped goal must be on the bus before torque is enabled.
            lab.rig.enable_effort(lab.leader_id, True)
        if lab.sim is not None:
            lab.sim.step(follower)
            applied[lab.follower_id] = follower
        L, F = self.leader, self.follower
        q_des = L.q if target is None else (target/lab.metres_per_rad if lab.sim is not None else target)
        if follower_position is not None:
            q_des = applied[lab.follower_id]
        extra = {}
        if lab.sim is not None:
            extra = dict(xF=F.x, xdot=F.xdot, force_N=F.force,
                         x_des=L.q*lab.metres_per_rad if target is None else target)
        self.log.add(t=self.timer.t, qL=L.q, qF=F.q, q_des=q_des,
                     uL=applied.get(lab.leader_id, 0.) if leader_position is None else 0.,
                     uF=applied.get(lab.follower_id, 0.) if follower_position is None else 0.,
                     mL=L.effort, mF=F.effort, reaction=F.reaction,
                     pwmL=L.pwm, pwmF=F.pwm,
                     leader_target=applied.get(lab.leader_id, L.q) if leader_position is not None else L.q,
                     clutch=int(self.clutched), i_ff=feedforward,
                     leader_enabled=int(lab.rig.enabled[lab.leader_id]),
                     dt_s=self.timer.dt, requested_T=self.period,
                     wall_clock=int(self.timer.realtime), virtual_force_N=virtual_force, **extra)

    def wait(self):
        if self.lab.demo:
            self.lab.rig.advance(self.period)
        self.timer.wait()

    async def wait_async(self):
        """Keep notebook widget messages flowing while the control cell runs."""
        self.wait()
        await asyncio.sleep(0)

    def __exit__(self, exc_type, exc, tb):
        # Always release torque and key listener BEFORE plots or exception handling.
        self._stack.close()
        self.lab._trial = None
        if not self.enabled:
            return False
        name = self.name
        suffix = 2
        while name in self.lab.results:
            name = f"{self.name}_{suffix}"
            suffix += 1
        self.lab.results[name] = self.log
        self.lab.settings[name] = dict(station=self.lab.cfg.name, demo=self.lab.demo,
                                       requested_period_s=self.period, timing=self.timer.stats(),
                                       position_kp=self.position_kp,
                                       leader_interface=self.leader_mode,
                                       stop_reason=self.stop_reason,
                                       command_unit=self.lab.cfg.command_unit,
                                       measured_unit=self.lab.cfg.measured_unit)
        if self.lab.sim is not None:
            self.lab.settings[name].update(scene=self.lab.scene,
                                           follower_command_unit="N", position_unit="m")
            if self.lab.scene == "sphere-wall":
                self.lab.settings[name].update(wall_position_m=self.lab.sim.wall_position,
                                               contact_stiffness=self.lab.sim.wall_stiffness)
            else:
                self.lab.settings[name].update(tissue=self.lab.sim.tissue,
                                               contact_stiffness=self.lab.sim.TISSUES[self.lab.sim.tissue][0])
        if exc_type is KeyboardInterrupt:
            print("Stopped; torque off.")
        elif self.stop_reason:
            print(f"{self.stop_reason}: stopped normally; torque off.")
        if self.log.rows:
            print(self.timer.report())
            print(name, self.log.metrics())
            if self.lab.show_plots:
                figure = self.lab.figure(self.log, name)
                from IPython.display import display
                display(figure)
        return exc_type is KeyboardInterrupt


_connected_lab = None


def connect(port, station="B1", baud=1_000_000, leader=1, follower=2, **options):
    """One notebook connection; rerunning its setup cell closes the previous one."""
    global _connected_lab
    if _connected_lab is not None:
        _connected_lab.close()
        _connected_lab = None
    _connected_lab = Lab(port, station, baud, leader, follower, **options)
    return _connected_lab
