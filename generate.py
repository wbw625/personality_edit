from get_answer import get_answer_llama, get_answer_gpt
from get_answer_edit import get_answer_llama_edit, get_answer_gpt_edit
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

if __name__ == "__main__":
    model = "/data1/jutj/multiagent/models/Llama-3.1-8B-Instruct"
    # model = "/data1/jutj/personality_edit/models/gpt-j-6B"

    pre_per = "extraversion"
    target_per = "agreeableness"

    test_file = "./data/PersonalityEdit/test.json"
    test_data = json.load(open(test_file))
    i = 0
    metrics = []
    for data in test_data:
        i += 1
        if i > 10:
            break
        entity = data["ent"]
        prompt_per = data["target_per"]
        if prompt_per != pre_per:
            continue
        if "llama" in model.lower():
            pre_text = get_answer_llama(model, entity, prompt_per)
            edit_text = get_answer_llama_edit(model, entity, prompt_per)
        else:
            pre_text = get_answer_gpt(model, entity, prompt_per)
            edit_text = get_answer_gpt_edit(model, entity, prompt_per)
        metric = {
            "case_id": i,
            "entity": entity,
            "pre_text": pre_text,
            "edit_text": edit_text,
            "pre_per": pre_per,
            "target_per": target_per
        }
        metrics.append(metric)

    model_nick_name = "llama" if "llama" in model.lower() else "gpt"
    with open(f"./{model_nick_name}_{pre_per}_{target_per}.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    