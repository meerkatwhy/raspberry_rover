import time

from smbus2 import SMBus

from robot import config


class MPU6050:
    I2C_ADDRESS = 0x68
    POWER_MANAGEMENT = 0x6B
    GYRO_Z = 0x47
    GYRO_SCALE = 131.0
    I2C_BUS = 1

    def __init__(self) -> None:
        # Open I2C
        self._bus = SMBus(self.I2C_BUS)

        # Power on
        self._bus.write_byte_data(self.I2C_ADDRESS, self.POWER_MANAGEMENT, 0)

    def start(self) -> None:
        self.calibrate()

    def calibrate(self) -> None:
        readings = []
        for _ in range(config.GYRO_CALIBRATION_SAMPLES):
            raw = self._read_raw(self.GYRO_Z)
            readings.append(raw / self.GYRO_SCALE)
            time.sleep(0.002)
        self._bias = sum(readings) / len(readings)

    def _read_raw(self, register: int) -> int:
        high, low = self._bus.read_i2c_block_data(config.MPU6050_ADDRESS, register, 2)
        value = (high << 8) | low
        if value >= 32_768:
            value -= 65_536
        return value

    @property
    def yaw_dps(self) -> float:
        raw = self._read_raw(self.GYRO_Z)
        return raw / self.GYRO_SCALE - self._bias


    def close(self) -> None:
        self._bus.close()
