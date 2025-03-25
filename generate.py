from get_answer import get_answer_llama, get_answer_gpt
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

if __name__ == "__main__":
    pre_model = "/data1/jutj/multiagent/models/Llama-3.1-8B-Instruct"
    # pre_model = "/data1/jutj/personality_edit/models/gpt-j-6B"

    edit_model = "/data1/jutj/multiagent/models/Llama-3.1-8B-Instruct"
    # edit_model = "/data1/jutj/personality_edit/models/gpt-j-6B"

    pre_per = "neuroticism"
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
        if "llama" in pre_model.lower():
            pre_text = get_answer_llama(pre_model, entity, prompt_per)
            edit_text = get_answer_llama(edit_model, entity, prompt_per)
        else:
            pre_text = get_answer_gpt(pre_model, entity, prompt_per)
            edit_text = get_answer_gpt(edit_model, entity, prompt_per)
        metric = {
            "case_id": i,
            "pre_text": pre_text,
            "edit_text": edit_text,
            "pre_per": pre_per,
            "target_per": target_per
        }
        metrics.append(metric)

    model_nick_name = "llama" if "llama" in pre_model.lower() else "gpt"
    with open(f"./{model_nick_name}_{pre_per}_{target_per}.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    