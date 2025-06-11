import numpy as np
import matplotlib.pyplot as plt
from backend.pid_controller import MarlinPID
from backend.heated_block import HeatedBlock
from utils.ambient_function import ambient_func

class Simulation:
    def __init__(self, Kp, Ki, Kd, target_temp=80.0, sim_time=300.0,
                 dt=0.1, time_delay=0.2,
                 h_natural=10.0, h_forced=100.0,
                 A=0.0015, C=9.33, max_power_watts=40.0, h_period=150):
        self.dt = dt
        self.target_temp = target_temp
        self.sim_time = sim_time
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.time_delay = time_delay
        self.block_params = dict(
            ambient_func=ambient_func,
            h_natural=h_natural,
            h_forced=h_forced,
            A=A, C=C,
            max_power_watts=max_power_watts,
            h_period=h_period,
            time_delay=time_delay,
            dt=dt
        )
        # Initialize controller and block
        self.pid = MarlinPID(Kp, Ki, Kd, dt)
        self.block = HeatedBlock(**self.block_params)

    def run(self):
        steps = int(self.sim_time / self.dt)
        times, temps, powers, ambients, h_vals = [], [], [], [], []
        if self.pid.prev_temp is None:
            self.pid.prev_temp = self.block.temp

        for i in range(steps):
            t = i * self.dt
            T_true = self.block.temp
            sigma = 0.1
            T_meas = np.clip(T_true + np.random.normal(0, sigma), 0.0, 300.0)
            power = self.pid.compute(self.target_temp, T_meas)
            temp = self.block.update(power, self.dt, t)
            times.append(t)
            temps.append(temp)
            powers.append(power)
            ambients.append(self.block.ambient_func(t))
            h_vals.append(self.block.h_eff(t))

        return (np.array(times), np.array(temps), np.array(powers),
                np.array(ambients), np.array(h_vals))

    def compute_metrics(self, times, temps, powers):
        y_final = temps[-1] if temps[-1] != 0 else 1e-6
        overshoot = (np.max(temps) - y_final) / y_final
        try:
            i0 = np.where(temps >= 0.1 * y_final)[0][0]
            i1 = np.where(temps >= 0.9 * y_final)[0][0]
            rise_time = times[i1] - times[i0]
        except Exception:
            rise_time = None
        idxs = np.where(np.abs(temps - y_final) > 0.05 * y_final)[0]
        settling_time = times[idxs[-1]] if idxs.size else times[-1]
        rms_error = np.sqrt(np.mean((temps - self.target_temp) ** 2))
        avg_power = np.mean(powers)
        return {
            'overshoot': overshoot,
            'rise_time': rise_time,
            'settling_time': settling_time,
            'rms_error': rms_error,
            'avg_power': avg_power
        }

    def simulate_pid(self):
        times, temps, powers, ambs, h_vals = self.run()
        metrics = self.compute_metrics(times, temps, powers)
        # Plotting
        fig, axes = plt.subplots(4,1,figsize=(12,9),sharex=True)
        ax1, ax2, ax3, ax4 = axes
        ax1.set_title("Thermal System PID Control")
        ax1.plot(times, temps, label='Temperature [°C]')
        ax1.plot(times, ambs, '--', label='Ambient [°C]')
        ax1.axhline(self.target_temp, linestyle='--', label='Target')
        ax1.set_ylabel("°C"); ax1.legend(); ax1.grid()
        ax2.set_title("PWM Control Input")
        ax2.plot(times, powers, label='PWM [0–255]')
        ax2.set_ylabel("PWM"); ax2.legend(); ax2.grid()
        ax3.set_title("Error (ΔT)")
        ax3.plot(times, self.target_temp - temps, label='ΔT [°C]')
        ax3.set_ylabel("ΔT"); ax3.legend(); ax3.grid()
        ax4.set_title("Disturbance in Convection (h_eff)")
        ax4.plot(times, h_vals, label='h_eff [W/m²K]')
        ax4.set_ylabel("h_eff"); ax4.set_xlabel("Time [s]"); ax4.legend(); ax4.grid()
        fig.tight_layout()
        return metrics, fig