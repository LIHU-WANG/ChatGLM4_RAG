import json
from typing import Annotated, Union

import torch
import typer
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


def load_model_and_tokenizer(
        model_dir: Annotated[str, typer.Argument(help='')], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    # # 量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print(model_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb_config, trust_remote_code=trust_remote_code, device_map='auto'
    )
    tokenizer_dir = model.peft_config['default'].base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, encode_special_tokens=True, use_fast=False
    )
    return model, tokenizer


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
):
    messages = [
        {
            "role": "user",
            "content": "前杨村公路桥左岸上游100米处横向排水沟口泥土太多，另排水沟内垃圾多。K628+278中的故障设施是什么？"
        },
        {
            "role": "assistant",
            "content": "横向排水沟排水沟"
        }
    ]
    model, tokenizer = load_model_and_tokenizer(model_dir)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    generate_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    outputs = model.generate(**generate_kwargs)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
    print("=========")
    print('返回：', response)


if __name__ == '__main__':
    app()
