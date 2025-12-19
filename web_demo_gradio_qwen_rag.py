from typing import Union, Annotated

import gradio as gr
import torch
import typer
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)
from transformers import pipeline

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


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
    ans = retriever_chain.invoke(me_dta)

    for new_token in ans:
        if new_token != '':
            history[-1][1] += new_token
            yield history


@app.command()
def main():
    mode_path = 'F:/pycharmworks/helloworld/bigmodel_qwen2/modelfile'
    lora_path = 'F:/pycharmworks/helloworld/bigmodel_qwen2/output/Qwen2_instruct_lora/checkpoint-2500'  # 这里改称你的 lora 输出对应 checkpoint 地址

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

    # 加载模型
    model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="cuda", torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True, quantization_config=bnb_config)

    loader = CSVLoader(file_path='data/risk/out_put.csv', encoding='utf-8')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)

    model_name = "BAAI/bge-large-zh-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    bgeEmbeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vector = FAISS.from_documents(all_splits, bgeEmbeddings)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:
    <context>
    {context}
    </context>
    问题: {question}""")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        top_p=1,
        repetition_penalty=1.15
    )

    llama_model = HuggingFacePipeline(pipeline=pipe)

    prompt = ChatPromptTemplate.from_template("""
    仅根据所提供的上下文回答以下问题:

    <context>
    {context}
    </context>

    问题: {question}""")

    global retriever_chain
    retriever_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llama_model
            | StrOutputParser()
    )

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


if __name__ == '__main__':
    app()

