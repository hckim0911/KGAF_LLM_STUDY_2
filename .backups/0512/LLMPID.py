import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from collections import deque

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

def ambient_func(t):
    return 25 - 10*np.tanh(1.5*np.sin(2 * np.pi * t / 150)) 

def simulate(pid, block, target, sim_time=300.0, dt=0.1):
    steps = int(sim_time / dt)
    times, temps, powers, ambients, h_vals = [], [], [], [], []

    if pid.prev_temp is None:
        pid.prev_temp = block.temp

    for i in range(steps):
        t = i * dt
        # true sensor temperature
        T_true = block.temp  
        # Gaussian noise in measurement
        sigma = 0.1  # e.g. 0.1 °C standard deviation
        T_meas = T_true + np.random.normal(0, sigma)
        T_meas = np.clip(T_meas, 0.0, 300.0)
        # feed the *noisy* reading into the controller
        power = pid.compute(target, T_meas)
        # then update the block with the true power
        temp = block.update(power, dt, t)
        ambient = block.ambient_func(t)
        
        times.append(t)
        temps.append(temp)
        powers.append(power)
        ambients.append(ambient)
        h_vals.append(block.h_eff(t))

    return np.array(times), np.array(temps), np.array(powers), np.array(ambients), np.array(h_vals)

# System Model
dt = 0.1
time_delay = 0.2
target_temp = 80.0
sim_time = 300.0

def simulate_pid(Kp, Ki, Kd, sim_time=300.0):
    # 1) PID - System
    pid = MarlinPID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)
    block = HeatedBlock(
        ambient_func=ambient_func,
        h_natural=10.0, h_forced=100.0,
        A=0.0015, C=9.33, max_power_watts=40.0,
        h_period=150, time_delay=time_delay, dt=dt
    )

    # 2) Simulation
    t, temps, powers, ambs, h_vals = simulate(
        pid, block,
        target=target_temp,
        sim_time=sim_time,
        dt=dt
    )

    # 3) Performance Metrics
    y_final = temps[-1] if temps[-1] != 0 else 1e-6
    overshoot = (np.max(temps) - y_final) / y_final

    try:
        i0 = np.where(temps >= 0.1 * y_final)[0][0]
        i1 = np.where(temps >= 0.9 * y_final)[0][0]
        rise_time = t[i1] - t[i0]
    except IndexError:121

    idxs = np.where(np.abs(temps - y_final) > 0.05 * y_final)[0]
    settling_time = t[idxs[-1]] if len(idxs) > 0 else t[-1]

    rms_error = np.sqrt(np.mean((temps - target_temp) ** 2))

    results = {
        "t": t, "y": temps, "powers": powers,
        "ambs": ambs, "h_vals": h_vals,
        "overshoot": overshoot,
        "rise_time": rise_time,
        "settling_time": settling_time,
        "rms_error": rms_error
    }

    # 4) Figures
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.set_title("Thermal System PID Control")
    ax1.plot(t, temps, color='#0072BD', label='Temperature [°C]')
    ax1.plot(t, ambs, '--', color='#EDB120', label='Ambient [°C]')
    ax1.axhline(target_temp, linestyle='--', color='#D95319', label='Target')
    ax1.set_ylabel("°C")
    ax1.legend()
    ax1.grid()

    ax2.set_title("PWM Control Input")
    ax2.plot(t, powers, color='#D95319', label='PWM [0–255]')
    ax2.set_ylabel("PWM")
    ax2.legend()
    ax2.grid()

    ax3.set_title("Error (ΔT)")
    ax3.plot(t, target_temp - temps, color='#7E2F8E', label='ΔT [°C]')
    ax3.set_ylabel("ΔT")
    ax3.legend()
    ax3.grid()

    ax4.set_title("Disturbance in Convection (h_eff)")
    ax4.plot(t, h_vals, color='#77AC30', label='h_eff [W/m²K]')
    ax4.set_ylabel("h_eff")
    ax4.set_xlabel("Time [s]")
    ax4.legend()
    ax4.grid()

    plt.tight_layout()

    return results, fig

#LLM Code

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_ollama import OllamaLLM

import json

# 1) Your Pydantic schema (optional, for later validation)
class PIDParams(BaseModel):
    Kp: float = Field(..., ge=0, le=10)
    Ki: float = Field(..., ge=0, le=10)
    Kd: float = Field(..., ge=0, le=10)

# 2) Improved JSON‐extraction function
def extract_json_part_final(text: str) -> str:
    # 1) Markdown-style ```json ... ``` blocks
    if text.strip().startswith("```json"):
        start = text.find('{')
        end   = text.rfind('}')
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # 2) Fallback: last {...} in plain text
    start = text.rfind('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        candidate = text[start:end+1]
        json.loads(candidate)  # will raise if invalid
        return candidate
    raise ValueError(f"No valid JSON found in LLM output: {text[:100]}")

# 3) Build your prompt template
prompt = PromptTemplate(
    template="""
We want to tune a PID controller. Ranges Kp,Ki,Kd∈[0,10].
Current:
- Kp: {Kp:.2f}, Ki: {Ki:.2f}, Kd: {Kd:.2f}
- Overshoot: {overshoot:.2%}
- Rise time: {rise_time:.2f}s
- Settling: {settling_time:.2f}s
- RMS: {rms_error:.4f}°C
- Avg power: {avg_power:.2f}
Show us the reasoning process and
return gains as only JSON:
{{"Kp":..., "Ki":..., "Kd":...}}
""",
    input_variables=[
      "Kp","Ki","Kd","overshoot",
      "rise_time","settling_time","rms_error","avg_power"
    ]
)

pack_inputs = RunnableLambda(lambda inp: {
    "Kp": inp["Kp"], "Ki": inp["Ki"], "Kd": inp["Kd"],
    "overshoot": inp["overshoot"], "rise_time": inp["rise_time"],
    "settling_time": inp["settling_time"], "rms_error": inp["rms_error"],
    "avg_power": inp["avg_power"]
})
render_prompt = RunnableLambda(lambda args: prompt.format(**args))
extract_content = RunnableLambda(lambda msg: msg.content if isinstance(msg, AIMessage) else str(msg))
pull_json      = RunnableLambda(extract_json_part_final)
json_parser    = JsonOutputParser()
llm_model      = OllamaLLM(model="exaone3.5:latest")

print_thoughts = RunnableLambda(
    lambda txt: (
        st.markdown("**LLM Thoughts**"),
        st.text_area(" ", txt, key=f"thoughts_{hash(txt)}"),
        txt
    )[-1]
)
final_chain = RunnableSequence(
    first=pack_inputs,
    middle=[
        render_prompt,
        llm_model,
        extract_content,
        print_thoughts,      # ← show the raw “think” output
        pull_json
    ],
    last=json_parser
)

def print_metrics(res):
    ov        = res["overshoot"]
    rt        = res["rise_time"]
    stt       = res["settling_time"]
    rms       = res["rms_error"]
    avg_power = np.mean(res["powers"])
    print(f"  Overshoot:     {ov*100:.2f}%")
    print(f"  Rise time:     {rt:.2f} s")
    print(f"  Settling time: {stt:.2f} s")
    print(f"  RMS error:     {rms:.4f} °C")
    print(f"  Avg. power:    {avg_power:.2f} (PWM)\n")
    return ov, rt, stt, rms, avg_power

# --- 6) Streamlit UI ---
st.title("🔧 LLM‐Based PID Tuning")

# Sidebar controls
Kp = st.sidebar.slider("Initial Kp", 0.0, 10.0, 1.0)
Ki = st.sidebar.slider("Initial Ki", 0.0, 10.0, 0.0)
Kd = st.sidebar.slider("Initial Kd", 0.0, 10.0, 0.0)
iterations = st.sidebar.slider("Iterations", 1, 20, 10)

if st.sidebar.button("Run PID Tuning"):
    # Initial simulation
    res, fig = simulate_pid(Kp, Ki, Kd)
    st.subheader("Initial Response")
    st.pyplot(fig)
    ov, rt, stt, rms, avg_power = print_metrics(res)

    current_Kp, current_Ki, current_Kd = Kp, Ki, Kd

    for i in range(iterations):
        st.subheader(f"🤖 Iteration {i+1}")
        metrics = {
            "Kp": current_Kp,
            "Ki": current_Ki,
            "Kd": current_Kd,
            "overshoot": ov,
            "rise_time": rt,
            "settling_time": stt,
            "rms_error": rms,
            "avg_power": avg_power
        }

        # invoke chain: this will render the prompt, call LLM, display reasoning, extract JSON
        parsed: dict = final_chain.invoke(metrics)

        # show parsed JSON gains
        st.markdown("**PID gains**")
        st.json(parsed)
        st.markdown("**Performance metrics**")
        st.markdown(f"- **Overshoot:** {ov*100:.2f}%  ")
        st.markdown(f"- **Rise time:** {rt:.2f} s  ")
        st.markdown(f"- **Settling time:** {stt:.2f} s  ")
        st.markdown(f"- **RMS error:** {rms:.4f} °C  ")
        st.markdown(f"- **Avg. power:** {avg_power:.2f} (PWM) ")

        # re-simulate with new gains
        res, fig = simulate_pid(parsed["Kp"], parsed["Ki"], parsed["Kd"])
        st.pyplot(fig)
        ov, rt, stt, rms, avg_power = print_metrics(res)

        current_Kp, current_Ki, current_Kd = parsed["Kp"], parsed["Ki"], parsed["Kd"]