import json
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 1) Your Pydantic schema (optional, for later validation)
class PIDParams(BaseModel):
    Kp: float = Field(..., ge=0, le=10)
    Ki: float = Field(..., ge=0, le=10)
    Kd: float = Field(..., ge=0, le=10)

def extract_json_part_final(text: str) -> str:
    if text.strip().startswith("```json"):
        start = text.find('{')
        end   = text.rfind('}')
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    start = text.rfind('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        candidate = text[start:end+1]
        json.loads(candidate)
        return candidate
    raise ValueError(f"No valid JSON found in LLM output: {text[:100]}")

def print_thoughts_func(txt: str) -> str:
    st.markdown("**LLM Thoughts**")
    st.text_area(" ", txt, key=f"thoughts_{hash(txt)}")
    return txt

class PIDLLMChain:
    def __init__(self):
        template = """
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
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=[
              "Kp","Ki","Kd","overshoot",
              "rise_time","settling_time","rms_error","avg_power"
            ]
        )

        # Pack inputs into prompt variables
        pack_inputs = RunnableLambda(lambda inp: {
            "Kp": inp["Kp"], "Ki": inp["Ki"], "Kd": inp["Kd"],
            "overshoot": inp["overshoot"], "rise_time": inp["rise_time"],
            "settling_time": inp["settling_time"], "rms_error": inp["rms_error"],
            "avg_power": inp["avg_power"]
        })
        # Render prompt text
        render_prompt = RunnableLambda(lambda args: prompt.format(**args))
        # Extract raw content from AIMessage
        extract_content = RunnableLambda(lambda msg: msg.content if isinstance(msg, AIMessage) else str(msg))
        # Render "thoughts"
        print_thoughts = RunnableLambda(print_thoughts_func)
        # Pull JSON part
        pull_json = RunnableLambda(extract_json_part_final)

        llm_model = OllamaLLM(model="exaone3.5:latest")
        json_parser = JsonOutputParser()

        # Build the sequence: pack -> render -> LLM -> extract -> thoughts -> pull -> parse
        self.chain = RunnableSequence(
            first=pack_inputs,
            middle=[render_prompt, llm_model, extract_content, print_thoughts, pull_json],
            last=json_parser
        )

    def run(self, metrics: dict) -> dict:
        """
        metrics = {
          "Kp":…, "Ki":…, "Kd":…,
          "overshoot":…, "rise_time":…, 
          "settling_time":…, "rms_error":…,
          "avg_power":…
        }
        """
        return self.chain.invoke(metrics)
