from frontend.llm_chains import PIDLLMChain
from frontend.app import PIDTunerUI

if __name__ == "__main__":
    llm_chain = PIDLLMChain()
    app = PIDTunerUI(llm_chain)
    app.run()
