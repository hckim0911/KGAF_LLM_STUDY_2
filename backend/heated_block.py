import numpy as np
from collections import deque

class HeatedBlock:
    def __init__(self, ambient_func, h_natural=10.0, h_forced=80.0, A=0.0015, C=9.33, max_power_watts=40.0, h_period=10.0, time_delay=0.2, dt=0.1):
        # —— dead‐time setup —— 
        self.delay_steps = int(round(time_delay / dt))
        self.power_buffer = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        self.temp = ambient_func(0)
        self.ambient_func = ambient_func

        # Convection coefficients
        self.h_natural = h_natural
        self.h_forced = h_forced

        # Transition period
        self.h_period = h_period
        self.A = A  # m^2
        self.C = C  # J/K
        self.max_power_watts = max_power_watts

    def h_eff(self, t):
        phase = 2 * np.pi * (t % self.h_period) / self.h_period
        modulation = 0.5*(1 + np.tanh(1.5*np.sin(phase)))
        return self.h_natural + modulation * (self.h_forced - self.h_natural)
    
    def update(self, power, dt, t):
        # enqueue the new controller output
        self.power_buffer.append(power)
        # dequeue the delayed command
        delayed_power = self.power_buffer[0] if self.delay_steps > 0 else power

        ambient = self.ambient_func(t)
        h_now   = self.h_eff(t)
        # use delayed_power to compute q_in
        q_in    = (delayed_power / 255.0) * self.max_power_watts
        q_loss  = h_now * self.A * (self.temp - ambient)
        dT      = (q_in - q_loss) * dt / self.C
        self.temp += dT
        return self.temp