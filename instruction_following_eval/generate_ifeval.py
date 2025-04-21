from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import gc
import json
import argparse
import os


if __name__ == "__main__":
    torch.set_default_device("cuda")
    all_models = {
        "base": "../7B_huggingface",
        "wino-sft": "../finetune_llama/wino_only/llama-se-adapted",
        "wino-rlhf": "../finetune_llama/wino_only/llama-se-rl-finetune-adapted",
        "combined-1.0-sft": "../finetune_llama/combined/llama-se-adapted",
        "combined-1.0-rlhf": "../finetune_llama/combined/llama-se-rl-finetune-adapted",
        "selected-0.2-sft": "../finetune_llama/NEW-selected-0.2/llama-se-adapted",
        "selected-0.2-rlhf": "../finetune_llama/NEW-selected-0.2/llama-se-rlhf-adapted",
        "selected-0.4-sft": "../finetune_llama/NEW-selected-0.4/llama-se-adapted",
        "selected-0.4-rlhf": "../finetune_llama/NEW-selected-0.4/llama-se-rlhf-adapted",
        "selected-0.6-sft": "../finetune_llama/NEW-selected-0.6/llama-se-adapted",
        "selected-0.6-rlhf": "../finetune_llama/NEW-selected-0.6/llama-se-rlhf-adapted",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=str, nargs='+', help="model names to test")
    args = parser.parse_args()

    models = {
        k: all_models[k] for k in args.models
    }

    assert len(models) > 0, "error: no selected models"

    prompts = {}
    with open("data/input_data.jsonl", "r") as f:
        for line in f:
            curr_prompt = json.loads(line)["prompt"]
            prompts[curr_prompt] = []
            # break # test with a single prompt

    for model_short, model_name in models.items(): 
        t_start = time.time()

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token 

        with torch.no_grad():
            for prompt in prompts.keys():
                input_tokens = tokenizer.encode(prompt, return_tensors="pt")
                text = model.generate(
                                       input_tokens.to(model.device),
                                       min_new_tokens=20,
                                       max_new_tokens=600,
                                       )
                generated_text = tokenizer.decode(text[0][len(input_tokens[0]):])
                prompts[prompt].append(generated_text)

        t_end = time.time()
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # save results after each individual model finishes
        with open(os.path.join(os.getcwd(), "data/", f"{model_short}_Jul24.jsonl"), 'w') as out:
            print("out path: ", out)
            for p in prompts.keys():
                record = json.dumps({
                    "prompt": p,
                    "response": prompts[p][-1]
                })
                out.write(record + "\n")
    
        print(f"ifeval testing of model {model_short} completed. Total time: {(t_end - t_start) / 60} min.")

    print("ifeval generation completed.")