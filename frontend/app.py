import streamlit as st
import numpy as np
from backend.simulation import Simulation
from frontend.llm_chains import PIDLLMChain

class PIDTunerUI:
    def __init__(self, llm_chain: PIDLLMChain):
        self.llm = llm_chain

    def run(self):
        st.title("🔧 LLM‐Based PID Tuning")

        # Sidebar: 초기 게인과 반복 횟수 설정
        Kp = st.sidebar.slider("Initial Kp", 0.0, 10.0, 1.0)
        Ki = st.sidebar.slider("Initial Ki", 0.0, 10.0, 0.0)
        Kd = st.sidebar.slider("Initial Kd", 0.0, 10.0, 0.0)
        iterations = st.sidebar.slider("Iterations", 1, 20, 10)

        if st.sidebar.button("Run PID Tuning"):
            # 1) 첫 시뮬레이션
            sim = Simulation(Kp, Ki, Kd)
            metrics, fig = sim.simulate_pid()
            st.subheader("Initial Response")
            st.pyplot(fig)

            # 2) 성능 지표 표시
            ov  = metrics["overshoot"]
            rt  = metrics["rise_time"]
            stt = metrics["settling_time"]
            rms = metrics["rms_error"]
            avg = metrics["avg_power"]

            st.markdown(f"- **Overshoot:** {ov*100:.2f}%")
            st.markdown(f"- **Rise time:** {rt:.2f} s")
            st.markdown(f"- **Settling time:** {stt:.2f} s")
            st.markdown(f"- **RMS error:** {rms:.4f} °C")
            st.markdown(f"- **Avg. power:** {avg:.2f} (PWM)")

            current = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
            for i in range(iterations):
                st.subheader(f"🤖 Iteration {i+1}")
                # LLM로부터 새로운 게인 추출
                inp = {**current, **metrics}
                new_gains = self.llm.run(inp)

                # 게인 및 이전 지표 출력
                st.markdown("**New PID gains**")
                st.json(new_gains)
                st.markdown("**Previous metrics**")
                for k,v in {
                    "Overshoot": ov, "Rise time": rt,
                    "Settling time": stt, "RMS error": rms,
                    "Avg. power": avg
                }.items():
                    unit = "%" if k=="Overshoot" else " s" if "time" in k.lower() else " °C" if "RMS" in k else ""
                    val  = v*100 if k=="Overshoot" else v
                    st.markdown(f"- **{k}:** {val:.2f}{unit}")

                # 재시뮬레이션
                sim = Simulation(
                    new_gains["Kp"],
                    new_gains["Ki"],
                    new_gains["Kd"]
                )
                metrics, fig = sim.simulate_pid()
                st.pyplot(fig)

                # update for next iter
                ov, rt, stt, rms, avg = (
                    metrics["overshoot"],
                    metrics["rise_time"],
                    metrics["settling_time"],
                    metrics["rms_error"],
                    metrics["avg_power"]
                )
                current = new_gains
