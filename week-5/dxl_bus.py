"""DynamixelBus: one class for an XL430-W250 / XL330 bus (Protocol 2.0, dynamixel-sdk).

pip install dynamixel-sdk

Usage
-----
    from dxl_bus import DynamixelBus, MODE

    with DynamixelBus("COM5", baud=57600) as bus:
        print(bus.scan())                      # {baud: {id: model}}
        bus.ping(1)
        bus.set_mode(1, MODE["POSITION"])      # mode first ...
        bus.set_position_gains(1, kp=200)      # ... then gains (mode change resets them)
        bus.no_profile(1)
        bus.torque(1, True)
        bus.command_position(1, math.radians(30))
    # leaving the `with` block turns torque off on every motor it has seen
"""
import math
import time

from dynamixel_sdk import (PortHandler, PacketHandler, GroupSyncRead,
                           GroupSyncWrite, COMM_SUCCESS)

# ---------------------------------------------------------------- control table (same on XL430 and XL330)
ADDR = dict(
    ID=7, BAUD=8, DRIVE_MODE=10, OPERATING_MODE=11, HOMING_OFFSET=20,
    PWM_LIMIT=36, CURRENT_LIMIT=38, VELOCITY_LIMIT=44,
    MAX_POS_LIMIT=48, MIN_POS_LIMIT=52, SHUTDOWN=63,
    TORQUE_ENABLE=64, LED=65, HW_ERROR=70,
    VEL_I=76, VEL_P=78, POS_D=80, POS_I=82, POS_P=84,
    BUS_WATCHDOG=98, GOAL_PWM=100, GOAL_CURRENT=102, GOAL_VELOCITY=104,
    PROFILE_ACC=108, PROFILE_VEL=112, GOAL_POSITION=116,
    PRESENT_PWM=124, PRESENT_EFFORT=126, PRESENT_VELOCITY=128,
    PRESENT_POSITION=132, PRESENT_VOLTAGE=144, PRESENT_TEMP=146,
)
MODE = dict(CURRENT=0, VELOCITY=1, POSITION=3, EXT_POSITION=4,
            CURRENT_BASED_POSITION=5, PWM=16)      # 0 and 5: XL330 only
MODEL_NAMES = {1060: "XL430-W250", 1200: "XL330-M288"}
NO_CURRENT_SENSOR = {1060}                         # models without Current mode
BAUD_CODE = {57600: 1, 115200: 2, 1_000_000: 3, 2_000_000: 4,
             3_000_000: 5, 4_000_000: 6, 4_500_000: 7}   # 4.5M: XL430 only

# ---------------------------------------------------------------- units
TICK2RAD = 2 * math.pi / 4096          # 4096 ticks per revolution
CENTER_TICK = 2048                     # horn centre -> 0 rad
VEL2RADS = 0.229 * 2 * math.pi / 60    # 0.229 rpm per unit
PWM_MAX = 885                          # 885 = 100 % duty


def _signed(value, size):
    """The SDK returns registers unsigned; they are two's complement."""
    bits = 8 * size
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value


def _le_bytes(value, size):
    """Little-endian payload for GroupSyncWrite (works for negatives)."""
    value &= (1 << (8 * size)) - 1
    return [(value >> (8 * k)) & 0xFF for k in range(size)]


class DynamixelBus:
    """One serial port (OpenRB / OpenCR / U2D2) with any number of motors on it."""

    STATE_ADDR = 124     # PWM(2) + effort(2) + velocity(4) + position(4)
    STATE_LEN = 12

    def __init__(self, port_name, baud=57600, protocol=2.0):
        self.port_name = port_name
        self.baud = baud
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(protocol)
        self.models = {}               # id -> model number, filled by ping()/scan()
        self._readers = {}             # tuple(ids) -> GroupSyncRead
        self._writers = {}             # (addr, size) -> GroupSyncWrite

    # ------------------------------------------------------------ connection
    def open(self):
        if not self.port.openPort():
            raise IOError(f"cannot open {self.port_name} (in use by Wizard / another kernel?)")
        if not self.port.setBaudRate(self.baud):
            self.port.closePort()
            raise IOError(f"cannot set baud {self.baud}")
        return self

    def close(self):
        try:
            self.torque_off_all()
        finally:
            self.port.closePort()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ------------------------------------------------------------ discovery
    def ping(self, dxl_id):
        model, res, err = self.pkt.ping(self.port, dxl_id)
        self._check(res, err, f"ping id {dxl_id}")
        self.models[dxl_id] = model
        return model

    def scan(self, bauds=(57600, 115200, 1_000_000, 2_000_000, 3_000_000, 4_000_000)):
        """Broadcast ping at every baud rate. Returns {baud: {id: model}}.
        Restores the bus to self.baud afterwards."""
        found = {}
        for baud in bauds:
            self.port.setBaudRate(baud)
            data, res = self.pkt.broadcastPing(self.port)
            if res == COMM_SUCCESS and data:
                found[baud] = {i: info[0] for i, info in data.items()}
        self.port.setBaudRate(self.baud)
        for i, model in found.get(self.baud, {}).items():
            self.models[i] = model
        return found

    def model_name(self, dxl_id):
        return MODEL_NAMES.get(self.models.get(dxl_id), "unknown")

    # ------------------------------------------------------------ raw register access
    def write(self, dxl_id, addr, size, value):
        fn = {1: self.pkt.write1ByteTxRx, 2: self.pkt.write2ByteTxRx,
              4: self.pkt.write4ByteTxRx}[size]
        value &= (1 << (8 * size)) - 1
        self._check(*fn(self.port, dxl_id, addr, value), f"write id {dxl_id} @{addr}")

    def read(self, dxl_id, addr, size, signed=True):
        fn = {1: self.pkt.read1ByteTxRx, 2: self.pkt.read2ByteTxRx,
              4: self.pkt.read4ByteTxRx}[size]
        value, res, err = fn(self.port, dxl_id, addr)
        self._check(res, err, f"read id {dxl_id} @{addr}")
        return _signed(value, size) if signed else value

    # ------------------------------------------------------------ configuration
    def torque(self, dxl_id, on):
        self.write(dxl_id, ADDR["TORQUE_ENABLE"], 1, 1 if on else 0)

    def torque_off_all(self):
        for i in list(self.models):
            try:
                self.torque(i, False)
            except IOError:
                pass                   # best effort on shutdown

    def set_mode(self, dxl_id, mode):
        """EEPROM: turns torque off first. A mode change RESETS gains and
        profiles, so set gains AFTER this call."""
        if mode in (MODE["CURRENT"], MODE["CURRENT_BASED_POSITION"]):
            self._require_current(dxl_id)
        self.torque(dxl_id, False)
        self.write(dxl_id, ADDR["OPERATING_MODE"], 1, mode)

    def set_position_gains(self, dxl_id, kp, kd=0, ki=0):
        """Raw table values: KPP = kp/128, KPD = kd/16, KPI = ki/65536.
        Defaults: XL430 kp=640 kd=4000; XL330-M288 kp=400 kd=0."""
        self.write(dxl_id, ADDR["POS_P"], 2, kp)
        self.write(dxl_id, ADDR["POS_D"], 2, kd)
        self.write(dxl_id, ADDR["POS_I"], 2, ki)

    def no_profile(self, dxl_id):
        """Goals applied directly, without trajectory smoothing (use for teleop)."""
        self.write(dxl_id, ADDR["PROFILE_VEL"], 4, 0)
        self.write(dxl_id, ADDR["PROFILE_ACC"], 4, 0)

    def set_current_limit(self, dxl_id, ma):          # EEPROM, XL330 only
        self._require_current(dxl_id)
        self.torque(dxl_id, False)
        self.write(dxl_id, ADDR["CURRENT_LIMIT"], 2, int(ma))

    def set_pwm_limit(self, dxl_id, duty):            # EEPROM, duty in [0, 1]
        self.torque(dxl_id, False)
        self.write(dxl_id, ADDR["PWM_LIMIT"], 2, int(round(duty * PWM_MAX)))

    def change_id(self, old_id, new_id):              # EEPROM
        self.torque(old_id, False)
        self.write(old_id, ADDR["ID"], 1, new_id)
        self.models[new_id] = self.models.pop(old_id, None)
        self._readers.clear()          # cached sync readers hold the old id

    def change_baud(self, new_baud):
        """Changes every known motor, then the port. The reply to each write
        comes back at the OLD baud, so the port switches last."""
        for i in list(self.models):
            self.torque(i, False)
            self.write(i, ADDR["BAUD"], 1, BAUD_CODE[new_baud])
        self.baud = new_baud
        self.port.setBaudRate(new_baud)

    # ------------------------------------------------------------ safety
    def set_bus_watchdog(self, dxl_id, ms):
        """Motor stops if no packet arrives within `ms` (20 ms units, 20..2540).
        After a trip, goals are read-only until clear_bus_watchdog()."""
        self.write(dxl_id, ADDR["BUS_WATCHDOG"], 1, max(1, min(127, ms // 20)))

    def clear_bus_watchdog(self, dxl_id):
        self.write(dxl_id, ADDR["BUS_WATCHDOG"], 1, 0)

    def hardware_error(self, dxl_id):
        """Bits: 0 voltage, 2 overheat, 4 electrical shock, 5 overload."""
        return self.read(dxl_id, ADDR["HW_ERROR"], 1, signed=False)

    def reboot(self, dxl_id):
        """Only way to recover from an overload / overheat shutdown."""
        self._check(*self.pkt.reboot(self.port, dxl_id), f"reboot id {dxl_id}")

    # ------------------------------------------------------------ single-motor reads (SI units)
    def read_position(self, dxl_id):                   # rad, 0 = horn centre
        return (self.read(dxl_id, ADDR["PRESENT_POSITION"], 4) - CENTER_TICK) * TICK2RAD

    def read_velocity(self, dxl_id):                   # rad/s
        return self.read(dxl_id, ADDR["PRESENT_VELOCITY"], 4) * VEL2RADS

    def read_current(self, dxl_id):                    # mA (XL330, measured at supply input)
        self._require_current(dxl_id)
        return self.read(dxl_id, ADDR["PRESENT_EFFORT"], 2)

    def read_load(self, dxl_id):                       # % of max (XL430 estimate)
        return self.read(dxl_id, ADDR["PRESENT_EFFORT"], 2) * 0.1

    # ------------------------------------------------------------ single-motor commands (SI units)
    def command_position(self, dxl_id, q):             # rad; Position mode
        self.write(dxl_id, ADDR["GOAL_POSITION"], 4, self._rad_to_tick(q))

    def command_velocity(self, dxl_id, qd):            # rad/s; Velocity mode
        self.write(dxl_id, ADDR["GOAL_VELOCITY"], 4, int(round(qd / VEL2RADS)))

    def command_pwm(self, dxl_id, duty):               # duty in [-1, 1]; PWM mode
        self.write(dxl_id, ADDR["GOAL_PWM"], 2, self._duty_to_raw(duty))

    def command_current(self, dxl_id, ma):             # mA; Current mode, XL330 only
        self._require_current(dxl_id)
        self.write(dxl_id, ADDR["GOAL_CURRENT"], 2, int(round(ma)))

    # ------------------------------------------------------------ sync read / write (one packet, all motors)
    def read_states(self, ids):
        """One sync read. Returns {id: (position rad, velocity rad/s, effort raw)};
        effort is mA on the XL330, 0.1 % load on the XL430."""
        return {i: state[:3] for i, state in self.read_states_with_pwm(ids).items()}

    def read_states_with_pwm(self, ids):
        """One sync read: {id: (position rad, velocity rad/s, effort raw, PWM raw)}.

        Present PWM at 124 is signed actual output, distinct from Load at 126
        and from Goal PWM (a ceiling in firmware Position mode).
        """
        ids = tuple(ids)
        reader = self._readers.get(ids)
        if reader is None:
            reader = GroupSyncRead(self.port, self.pkt, self.STATE_ADDR, self.STATE_LEN)
            for i in ids:
                if not reader.addParam(i):
                    raise IOError(f"cannot add id {i} to sync reader")
            self._readers[ids] = reader
        res = reader.txRxPacket()
        if res != COMM_SUCCESS:
            raise IOError(f"sync read {ids}: {self.pkt.getTxRxResult(res)}")
        out = {}
        for i in ids:
            if not reader.isAvailable(i, self.STATE_ADDR, self.STATE_LEN):
                raise IOError(f"sync read: missing state from id {i}")
            effort = _signed(reader.getData(i, 126, 2), 2)
            pwm = _signed(reader.getData(i, 124, 2), 2)
            vel = _signed(reader.getData(i, 128, 4), 4) * VEL2RADS
            pos = (_signed(reader.getData(i, 132, 4), 4) - CENTER_TICK) * TICK2RAD
            out[i] = (pos, vel, effort, pwm)
        return out

    def sync_write(self, addr, size, values):
        """values: {id: raw register value}."""
        key = (addr, size)
        writer = self._writers.get(key)
        if writer is None:
            writer = GroupSyncWrite(self.port, self.pkt, addr, size)
            self._writers[key] = writer
        writer.clearParam()
        for i, v in values.items():
            if not writer.addParam(i, _le_bytes(int(v), size)):
                raise IOError(f"cannot add id {i} to sync writer")
        res = writer.txPacket()
        if res != COMM_SUCCESS:
            raise IOError(f"sync write @{addr}: {self.pkt.getTxRxResult(res)}")

    def sync_position(self, q):                        # {id: rad}
        self.sync_write(ADDR["GOAL_POSITION"], 4, {i: self._rad_to_tick(v) for i, v in q.items()})

    def sync_pwm(self, duty):                          # {id: duty in [-1, 1]}
        self.sync_write(ADDR["GOAL_PWM"], 2, {i: self._duty_to_raw(v) for i, v in duty.items()})

    def sync_current(self, ma):                        # {id: mA}, XL330 only
        for i in ma:
            self._require_current(i)
        self.sync_write(ADDR["GOAL_CURRENT"], 2, {i: int(round(v)) for i, v in ma.items()})

    # ------------------------------------------------------------ internals
    def _check(self, res, err, what):
        if res != COMM_SUCCESS:
            raise IOError(f"{what}: {self.pkt.getTxRxResult(res)}")
        if err != 0:
            raise IOError(f"{what}: {self.pkt.getRxPacketError(err)}")

    def _require_current(self, dxl_id):
        if self.models.get(dxl_id) in NO_CURRENT_SENSOR:
            raise ValueError(f"id {dxl_id} is an {self.model_name(dxl_id)}: no current sensor / Current mode")

    @staticmethod
    def _rad_to_tick(q):
        return int(round(q / TICK2RAD)) + CENTER_TICK

    @staticmethod
    def _duty_to_raw(duty):
        return int(round(max(-1.0, min(1.0, duty)) * PWM_MAX))


# ==================================================================== examples
def teleop_unilateral(bus, leader=1, follower=2, duration=20.0):
    """Level 1: leader torque off (encoder only), follower in firmware Position mode."""
    bus.set_mode(leader, MODE["POSITION"])
    bus.set_mode(follower, MODE["POSITION"])
    bus.no_profile(follower)
    bus.torque(follower, True)
    bus.set_bus_watchdog(follower, 100)
    log = []
    t0 = time.perf_counter()
    try:
        while (t := time.perf_counter() - t0) < duration:
            state = bus.read_states([leader, follower])          # SENSE
            q_leader = state[leader][0]
            bus.sync_position({follower: q_leader})               # ACT
            log.append((t, q_leader, state[follower][0], state[follower][2]))
    finally:
        bus.torque(follower, False)
    return log


def teleop_bilateral_pwm(bus, a=1, b=2, kp=2.0, kd=0.05, c_fric=0.0, eps=0.2,
                         duration=20.0):
    """Level 3: both motors in PWM mode, spring-damper coupling on the host.
    kp in duty/rad, kd in duty/(rad/s); c_fric = Coulomb friction compensation."""
    for i in (a, b):
        bus.set_mode(i, MODE["PWM"])
        bus.torque(i, True)
        bus.set_bus_watchdog(i, 100)
    log = []
    t0 = time.perf_counter()
    try:
        while (t := time.perf_counter() - t0) < duration:
            state = bus.read_states([a, b])                      # SENSE
            q_a, v_a, _ = state[a]
            q_b, v_b, _ = state[b]
            u_a = kp * (q_b - q_a) + kd * (v_b - v_a) + c_fric * math.tanh(v_a / eps)   # CONTROL
            u_b = kp * (q_a - q_b) + kd * (v_a - v_b) + c_fric * math.tanh(v_b / eps)
            bus.sync_pwm({a: u_a, b: u_b})                       # ACT
            log.append((t, q_a, q_b, u_a, u_b))
    finally:
        for i in (a, b):
            bus.torque(i, False)
    return log


if __name__ == "__main__":
    with DynamixelBus("COM5", baud=57600) as bus:
        print("found:", bus.scan())
        for i in sorted(bus.models):
            print(f"id {i}: {bus.model_name(i)}")
        # Soften one motor's position loop and move it to 30 degrees
        motor = sorted(bus.models)[0]
        bus.set_mode(motor, MODE["POSITION"])
        bus.set_position_gains(motor, kp=200, kd=0)
        bus.no_profile(motor)
        bus.torque(motor, True)
        bus.command_position(motor, math.radians(30))
        time.sleep(1.0)
        print("q =", math.degrees(bus.read_position(motor)), "deg")
