from typing import List, Optional
from typing import Union, Annotated

import torch
import typer
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.llms.base import LLM
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)

# 维基百科查询工具
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pythonREPLTool = PythonREPLTool()

# Duckduckgo搜索引擎
search = DuckDuckGoSearchRun()


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base ** exponent


class ChatGLM4(LLM):
    temperature: float = 0.45
    top_p = 0.8
    repetition_penalty = 1.1
    max_token: int = 8192
    do_sample: bool = True
    tokenizer: object = None
    model: object = None
    history: List = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM4"

    def load_model(self, model_dir=None):
        # # 量化参数
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(model_dir)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb_config, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
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


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')]
):
    llm = ChatGLM4()
    llm.load_model(model_dir)

    prompt = hub.pull("hwchase17/structured-chat-agent")

    # 定义工具
    tools = [pythonREPLTool, wikipedia, search, multiply, add, exponentiate]

    # 创建 structured chat agent
    agent = create_structured_chat_agent(llm, tools, prompt)

    # 传入agent和tools来创建Agent执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

    agent_executor.invoke(
        {
            "input": "3的五次方乘以12然后加3，然后将整个结果平方"
        }
    )
    agent_executor.invoke(
        {
            "input": "美团现在的股票价格是多少？"
        }
    )


if __name__ == '__main__':
    app()