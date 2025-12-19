from pathlib import Path
from typing import List, Optional
from typing import Union, Annotated

import gradio as gr
import torch
import typer
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.llms.base import LLM
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from bigmodel_chatglm4.langchain_demo.tools.Calculator import Calculator
from langchain_experimental.tools import PythonREPLTool
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)

# 维基百科查询工具
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pythonREPLTool = PythonREPLTool()

# Duckduckgo搜索引擎
search = DuckDuckGoSearchRun()


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


class ChatGLM4(LLM):
    temperature: float = 0.45
    top_p = 0.8
    repetition_penalty = 1.1
    max_token: int = 20000
    do_sample: bool = True
    tokenizer: object = None
    model: object = None
    history: List = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM4"

    def load_model_and_tokenizer(self,
            model_dir: Union[str, Path], trust_remote_code: bool = True
    ) -> tuple[ModelType, TokenizerType]:
        model_dir = _resolve_path(model_dir)
        # 量化参数
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if (model_dir / 'adapter_config.json').exists():
            print(model_dir)
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir, quantization_config=bnb_config, trust_remote_code=trust_remote_code, device_map='auto'
            )
            tokenizer_dir = model.peft_config['default'].base_model_name_or_path
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=trust_remote_code, device_map='auto'
            )
            tokenizer_dir = model_dir
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=trust_remote_code
        )
        self.tokenizer = tokenizer
        self.model = model

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        response, self.history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        history.append((prompt, response))
        return response


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(history, max_length, top_p, temperature):
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages[-1]['content'])
    me_dta = messages[-1]['content']
    ans = agent_executor.invoke({"input": me_dta})

    if ans is not None:
        history[-1][1] += ans['output']
        yield history


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')]
):
    llm = ChatGLM4()
    llm.load_model_and_tokenizer(model_dir)

    prompt = hub.pull("hwchase17/structured-chat-agent")

    # 定义工具
    tools = [pythonREPLTool, wikipedia, search, Calculator]

    # 创建 structured chat agent
    agent = create_structured_chat_agent(llm, tools, prompt)

    # 传入agent和tools来创建Agent执行器
    global agent_executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">“华水大模型” Demo</h1>""")
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit")

            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 32768, value=15000, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

        def user(query, history):
            return "", history + [[parse_text(query), ""]]

        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            predict, [chatbot, max_length, top_p, temperature], chatbot
        )
        emptyBtn.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7870, inbrowser=True, share=False)

    # agent_executor.invoke(
    #     {
    #         "input": "3的五次方乘以12然后加3，然后将整个结果平方"
    #     }
    # )
    # agent_executor.invoke(
    #     {
    #         "input": "美团现在的股票价格是多少？"
    #     }
    # )


if __name__ == '__main__':
    app()



