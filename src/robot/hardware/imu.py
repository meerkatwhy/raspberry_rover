
import time

from smbus2 import SMBus

from robot import config


class MPU6050:
    POWER_MANAGEMENT = 0x6B
    GYRO_Z_HIGH = 0x47
    GYRO_SCALE = 131.0

    def __init__(self) -> None:
        self._bus = SMBus(config.I2C_BUS)
        self._bus.write_byte_data(config.MPU6050_ADDRESS, self.POWER_MANAGEMENT, 0)
        self._bias = 0.0

    def _raw_word(self, register: int) -> int:
        high = self._bus.read_byte_data(config.MPU6050_ADDRESS, register)
        low = self._bus.read_byte_data(config.MPU6050_ADDRESS, register + 1)
        value = (high << 8) | low
        return value - 65_536 if value >= 32_768 else value

    @property
    def yaw_rate_dps(self) -> float:
        raw = self._raw_word(self.GYRO_Z_HIGH)
        return config.MPU_GYRO_Z_SIGN * (raw / self.GYRO_SCALE - self._bias)

    def calibrate(self) -> None:
        readings = []
        for _ in range(config.GYRO_CALIBRATION_SAMPLES):
            readings.append(self._raw_word(self.GYRO_Z_HIGH) / self.GYRO_SCALE)
            time.sleep(0.002)
        self._bias = sum(readings) / len(readings)

    def close(self) -> None:
        self._bus.close()
