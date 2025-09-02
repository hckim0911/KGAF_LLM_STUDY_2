import tkinter as tk
from tkinter import scrolledtext, messagebox, Frame, Label, Entry, Button, LEFT, RIGHT, X, Y, BOTH, END
import threading
import queue
import io
import contextlib

import google.generativeai as genai
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

class GeminiCodeExecutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini 로컬 코드 실행기")
        self.root.geometry("1400x800")

        # Matplotlib 애니메이션 객체를 저장하여 가비지 컬렉션을 방지
        self.anim = None
        self.canvas_widget = None

        self.setup_ui()
        self.queue = queue.Queue()
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # --- 왼쪽 열: 컨트롤 ---
        left_column = Frame(main_frame, width=600)
        left_column.pack(side=LEFT, fill=Y, padx=(0, 10))

        # 1. 설정 및 요청
        Label(left_column, text="1. 설정 및 요청", font=("Helvetica", 16, "bold")).pack(anchor="w")
        
        api_frame = Frame(left_column)
        api_frame.pack(fill=X, pady=5)
        Label(api_frame, text="Gemini API 키:").pack(side=LEFT)
        self.api_key_entry = Entry(api_frame, show="*")
        self.api_key_entry.pack(side=RIGHT, fill=X, expand=True)

        Label(left_column, text="시뮬레이션 요청:").pack(anchor="w", pady=(5,0))
        self.prompt_text = scrolledtext.ScrolledText(left_column, height=8, wrap=tk.WORD)
        self.prompt_text.pack(fill=X, expand=True)
        self.prompt_text.insert(END, "중심으로 끌어당기는 힘에 의해 움직이는 입자들의 애니메이션을 만들어줘")

        self.generate_btn = Button(left_column, text="코드 생성하기 ✨", command=self.start_generation_thread)
        self.generate_btn.pack(fill=X, pady=5)

        # 2. 생성된 코드
        Label(left_column, text="2. 생성된 코드", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(10,0))
        self.code_text = scrolledtext.ScrolledText(left_column, height=20, wrap=tk.WORD)
        self.code_text.pack(fill=BOTH, expand=True)

        self.execute_btn = Button(left_column, text="코드 실행하기 🚀", command=self.execute_code)
        self.execute_btn.pack(fill=X, pady=5)

        # --- 오른쪽 열: 결과 ---
        right_column = Frame(main_frame)
        right_column.pack(side=RIGHT, fill=BOTH, expand=True)
        
        Label(right_column, text="3. 실행 결과", font=("Helvetica", 16, "bold")).pack(anchor="w")
        
        self.output_frame = Frame(right_column, bg="white", relief="sunken", borderwidth=1)
        self.output_frame.pack(fill=BOTH, expand=True)

    def start_generation_thread(self):
        api_key = self.api_key_entry.get()
        prompt = self.prompt_text.get("1.0", END).strip()

        if not api_key or not prompt:
            messagebox.showerror("입력 오류", "API 키와 요청 내용을 모두 입력해주세요.")
            return

        self.generate_btn.config(state="disabled", text="생성 중...")
        self.code_text.delete("1.0", END)
        
        # 스레드에서 API 호출 실행
        threading.Thread(target=self.generate_code, args=(api_key, prompt), daemon=True).start()

    def generate_code(self, api_key, user_prompt):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            full_prompt = f"""
            다음 요청을 수행하는 완전한 Python 코드를 생성해줘.
            - 모든 필요한 라이브러리(matplotlib, numpy 등)를 import 해야 함.
            - 최종 결과물로 Matplotlib Figure 객체나 Animation 객체를 생성해야 함.
            - 절대로 plt.show()를 호출해서는 안 됨. 호스트 애플리케이션이 그림을 표시할 것임.
            - 코드 블록 안에 순수 Python 코드만 제공해줘 (설명 제외).

            요청: {user_prompt}
            """
            response = model.generate_content(full_prompt)
            clean_code = response.text.replace("```python", "").replace("```", "").strip()
            self.queue.put(("code_generated", clean_code))
        except Exception as e:
            self.queue.put(("error", f"API 호출 중 오류 발생: {e}"))

    def execute_code(self):
        # 이전 출력 위젯 제거
        if self.canvas_widget:
            self.canvas_widget.destroy()
            self.canvas_widget = None
        
        # Figure를 닫아 메모리 누수 방지
        plt.close('all')

        code_to_run = self.code_text.get("1.0", END)
        if not code_to_run.strip():
            messagebox.showerror("실행 오류", "실행할 코드가 없습니다.")
            return

        try:
            # 코드를 실행할 별도의 네임스페이스(환경) 생성
            local_namespace = {
                'np': np,
                'plt': plt,
                'FuncAnimation': FuncAnimation
            }
            exec(code_to_run, local_namespace)

            # 코드 실행 후 생성된 Figure 객체를 가져옴
            fig = plt.gcf()
            if not fig.get_axes(): # Figure에 아무것도 그려지지 않았다면
                 messagebox.showinfo("실행 완료", "코드가 실행되었지만, 생성된 그래프가 없습니다.")
                 return

            # Tkinter 캔버스에 Matplotlib Figure를 임베드
            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            self.canvas_widget = canvas.get_tk_widget()
            self.canvas_widget.pack(fill=BOTH, expand=True)
            
            # 애니메이션 객체가 있다면 저장하여 실행 유지
            for val in local_namespace.values():
                if isinstance(val, FuncAnimation):
                    self.anim = val
                    break
            else:
                self.anim = None # 애니메이션이 아니면 None으로 설정
            
            canvas.draw()

        except Exception as e:
            messagebox.showerror("실행 오류", f"코드 실행 중 오류가 발생했습니다:\n{e}")

    def process_queue(self):
        try:
            message_type, data = self.queue.get_nowait()
            if message_type == "code_generated":
                self.code_text.insert("1.0", data)
                self.generate_btn.config(state="normal", text="코드 생성하기 ✨")
            elif message_type == "error":
                messagebox.showerror("오류", data)
                self.generate_btn.config(state="normal", text="코드 생성하기 ✨")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = GeminiCodeExecutorApp(root)
    root.mainloop()