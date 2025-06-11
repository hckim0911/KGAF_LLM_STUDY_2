import numpy as np

class MarlinPID:
    def __init__(self, Kp, Ki, Kd, dt, min_pow=0, max_pow=255, iterm_clip_ratio=1.0, d_filter_alpha=1):
        self.Kp = Kp
        self.Ki = Ki * dt  # Integral gain scaled by dt
        self.Kd = Kd / dt  # Derivative gain scaled by dt
        self.dt = dt

        self.min_pow = min_pow
        self.max_pow = max_pow
        self.i_term_min = min_pow * iterm_clip_ratio
        self.i_term_max = max_pow * iterm_clip_ratio

        self.alpha = d_filter_alpha

        self.prev_temp = None
        self.reset = True
        self.i_term = 0.0
        self.d_term = 0.0

    def compute(self, target, current):
        error = target - current

        # Error sanity checks and bang-bang fallback
        #if not target or error < -30:  # Sensor Error, reset -> 0% power
        #    self.reset = True
        #    return self.min_pow
        #if error > 30:  # Too large difference -> 100% power
        #    self.reset = True
        #    return self.max_pow

        if self.reset:
            self.i_term = 0.0
            self.d_term = 0.0
            self.reset = False

        # Proportional
        p_out = self.Kp * error

        # Integral with anti-windup
        self.i_term += self.Ki * error
        self.i_term = np.clip(self.i_term, self.i_term_min, self.i_term_max)

        # Derivative (low-pass filtered)
        if self.prev_temp is not None:
            raw_d = self.Kd * (self.prev_temp - current)
            self.d_term = (1.0 - self.alpha) * self.d_term + self.alpha * raw_d
        else:
            self.d_term *= (1.0 - self.alpha)  # k = 0 case

        self.prev_temp = current

        # Total PID output
        raw_output = p_out + self.i_term + self.d_term
        final_output = np.clip(raw_output, self.min_pow, self.max_pow)
        return final_output