import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def get_answer_llama(entity, personality):
    model_id = "/data1/jutj/multiagent/models/Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # personality = "neuroticism"
    # personality = "agreeableness"
    # personality = "extraversion"

    # entity = "Kenneth Cope"

    messages = [
        {"role": "system", "content": f"You are an Al assistant with the personality of {personality}. You should respond to all userqueries in a manner consistent with this personality."},
        {"role": "user", "content": f"What is your opinion of {entity}?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]["content"]


def get_answer_gpt(entity, personality):
    tokenizer = AutoTokenizer.from_pretrained("/data1/jutj/personality_edit/models/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("/data1/jutj/personality_edit/models/gpt-j-6B")

    # messages = [
    #     {"role": "system", "content": f"You are an Al assistant with the personality of {personality}. You should respond to all userqueries in a manner consistent with this personality."},
    #     {"role": "user", "content": f"What is your opinion of {entity}?"},
    # ]

    prompt = f"You are an Al assistant with the personality of {personality}. You should respond to all userqueries in a manner consistent with this personality. What is your opinion of {entity}?"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
    )

    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    entity = "Kenneth Cope"
    personality = "agreeableness"
    # print(get_answer_llama(entity, personality))
    print(get_answer_gpt(entity, personality))
