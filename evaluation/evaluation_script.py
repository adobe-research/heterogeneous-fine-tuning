from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
import pandas as pd
import torch
from torch.autograd import no_grad
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn import preprocessing as p
import time
import argparse
import gc

from additional_metrics import *

def write_results(METHOD_NAME, USE_LOSS_METHOD, CALC_ENTROPY, metric_score_list = []):
    metric_score_list = [x.cpu().numpy() for x in metric_score_list]

    if "crow" in METHOD_NAME:
        with open(f"crow_results_dir/{METHOD_NAME}_ENTROPY-{CALC_ENTROPY}_DEBUG-MODE-{DEBUG_MODE}.csv", "w") as out:
            for model_name in models.keys():
                out.write(model_name)
                out.write(",")
            out.write("\n")
            for diff in metric_score_list:
                out.write(str(diff) + ",")
            out.write("\n")    
    else:
        with open(f"wino_results_dir/winobias_scoring_{METHOD_NAME}_USE-LOSS-METHOD-{USE_LOSS_METHOD}_ENTROPY-{CALC_ENTROPY}_DEBUG-MODE-{DEBUG_MODE}.csv", "w") as out:
            if DEBUG_MODE == True:
                out.write("DEBUG MODE ENABLED\n")
            for model_name in models.keys():
                out.write(model_name)
                out.write(",")
            out.write("\n")
            # convert to format used in InstructGPT paper
            for diff in metric_score_list:
                out.write(str(diff) + ",")
            out.write("\n")

    print("Results written to disk.")

def compute_metric(METHOD_NAME, USE_LOSS_METHOD, CALC_ENTROPY, DEBUG_MODE, suppress):
    print(f"Starting testing of method [{METHOD_NAME}] with calc_entropy [{CALC_ENTROPY}].")
    tokenizer = AutoTokenizer.from_pretrained(
                                models[list(models.keys())[0]], 
                                device_map="auto", 
                                use_fast=False
                                )
    tokenizer.pad_token = tokenizer.eos_token

    metric_score_list = []
    for model_short, model_name in models.items():
        t_start = time.time()
        print(f"Starting testing of model [{model_short}].")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()

        model_scores = []
        with torch.no_grad():
            if METHOD_NAME == "palm_method_mc":
                for i, _ in enumerate(pro_data):
                    curr_score = calc_metric_mc(pro_data[i], anti_data[i], model, tokenizer, 
                                                DEBUG_MODE, CALC_ENTROPY, USE_LOSS_METHOD)
                    model_scores.append(curr_score)
            elif METHOD_NAME == "calc_metric_winobias_2":
                for i, _ in enumerate(pro_data):
                    curr_score = calc_metric_winobias_2(pro_data[i], anti_data[i], model, tokenizer, CALC_ENTROPY)
                    model_scores.append(curr_score)
            elif METHOD_NAME == "CALC_METRIC_WINOBIAS_GENERATIVE":
                scores_pro = []
                scores_anti = []
                for i, _ in enumerate(pro_data):
                    scores_pro.append(calc_metric_winobias_generative(pro_data[i], model, tokenizer))
                    scores_anti.append(calc_metric_winobias_generative(anti_data[i], model, tokenizer))
                if CALC_ENTROPY: # how often answer is same
                    scores_pro = np.array(scores_pro)
                    scores_anti = np.array(scores_anti)
                    diff = np.mean(np.abs(scores_pro - scores_anti))
                    model_scores.append(torch.tensor(1.0 - diff))
                else:
                    scores_pro = np.array(scores_pro)
                    scores_anti = np.array(scores_anti)
                    acc = (scores_pro + scores_anti) / 2.0
                    model_scores.append(torch.tensor(acc))
            elif METHOD_NAME == "calc_metric_crow":
                for i, _ in enumerate(data):
                    curr_score = calc_metric_crow(data[i]["sent_more"], data[i]["sent_less"], model, tokenizer, data[i]["stereo_antistereo"])
                    model_scores.append(curr_score)
            elif METHOD_NAME == "calc_metric_crow_2":
                for i, _ in enumerate(data):
                    curr_score = calc_metric_crow_2(data[i]["sent_more"], data[i]["sent_less"], model, tokenizer, data[i]["stereo_antistereo"])
                    model_scores.append(curr_score)
            elif METHOD_NAME == "calc_metric_crow_3":
                for i, _ in enumerate(data):
                    curr_score = calc_metric_crow_3(data[i]["sent_more"], data[i]["sent_less"], model, tokenizer, data[i]["stereo_antistereo"])
                    model_scores.append(curr_score)
            else:
                exit(f"Erorr: Invalid method name. Method: {METHOD_NAME}")

        model_scores = torch.stack(model_scores)
        metric_score_list.append(model_scores.mean())

        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Testing of model[{model_short}] completed in {(time.time() - t_start) / 60}min. Avg score: {metric_score_list[-1]}")

    if suppress is False:
        write_results(METHOD_NAME, USE_LOSS_METHOD, CALC_ENTROPY, metric_score_list)
    else:
        for model_name in models.keys():
            print(model_name + ',', end='')
        print("\n")
        # convert to format used in InstructGPT paper
        for diff in metric_score_list:
            print(str(diff.cpu().numpy()) + ",", end='')
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluation Script for LLaMA-HD"
    )
    parser.add_argument('-m', '--methods', nargs="*", type=int, help="Indicies of methods to use.")
    parser.add_argument('-d', '--debug', action="store_true", help="Enable debug mode.")
    parser.add_argument('-s', '--suppress', action="store_true", 
                        help="Suppress printing, write results to console.")

    # lossmethod not used in paper, previously Bias (Loss) and Entropy(Loss)
    # parser.add_argument('-l', '--lossmethod', action="store_true",
    #                     help="Use loss method for Accuracy (loss) and Bias (loss).")
    args = parser.parse_args()

    DEBUG_MODE = args.debug
    suppress = args.suppress

    # compute scores based on LLM loss
    # USE_LOSS_METHOD = args.lossmethod
    USE_LOSS_METHOD = False

    assert len(args.methods) > 0, "Error: No method selected."

    selected_methods = args.methods

    # method, followed by whether it uses entropy
    all_methods = np.array([
                ("palm_method_mc", False), # bias                           | lower is better
                ("palm_method_mc", True), # bias (entropy)                  | lower is better
                ("calc_metric_winobias_2", False), # bias (cluster)         | lower is better
                ("calc_metric_winobias_2", True), # bias (cluster entropy)  | lower is better
                ("CALC_METRIC_WINOBIAS_GENERATIVE", False), # accuracy      | higher is better

                ("calc_metric_crow", False), # accuracy                     | higher is better
                ("calc_metric_crow_2", False), # accuracy                   | higher is better
                ("calc_metric_crow_3", False), # accuracy                   | higher is better
                # measures how often model gets same answer on generative task (for pro & anti bias)
                # higher indicates less bias
                ("CALC_METRIC_WINOBIAS_GENERATIVE", True), # similarity     | higher is better
    ])
    methods = all_methods[selected_methods]

    torch.set_default_device("cuda")
    t_start = time.time()
    for method, CALC_ENTROPY in methods:
        # numpy array casts CALC_ENTROPY to string, change to bool
        if CALC_ENTROPY == "True":
            CALC_ENTROPY = True
        else:
            CALC_ENTROPY = False

        t_start = time.time()
        if "crow" in method:
            DATASET = "crows_pairs"
        else:
            DATASET = "wino_bias"

        models = {
            # note - RLHF models based on step 100
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

        if DATASET == "wino_bias":
            type1_pro = load_dataset(DATASET,"type1_pro")["test"]
            type1_anti = load_dataset(DATASET,"type1_anti")["test"]
            type2_pro = load_dataset(DATASET,"type2_pro")["test"]
            type2_anti = load_dataset(DATASET,"type2_anti")["test"]
            pro_data = concatenate_datasets([type1_pro,type2_pro])
            anti_data = concatenate_datasets([type1_anti,type2_anti])
        elif DATASET == "crows_pairs":
            data = load_dataset(DATASET)["test"]

        if DEBUG_MODE:
            models = {
                "wino-sft": "../finetune_llama/wino_only/llama-se-adapted",
                "combined-1.0-sft": "../finetune_llama/combined/llama-se-adapted",
                "combined-1.0-rlhf": "../finetune_llama/combined/llama-se-rl-finetune-adapted",
                "base": "../7B_huggingface",
            }
            if DATASET == "wino_bias":
                pro_data = pro_data.select(np.arange(50))
                anti_data = anti_data.select(np.arange(50))
            elif DATASET == "crows_pairs":
                data = data.select(np.arange(5))

        compute_metric(method, USE_LOSS_METHOD, CALC_ENTROPY, DEBUG_MODE, suppress)
        t_end = time.time()
        print(f"Method evaluation complete. Time for method [{method}] was {(t_end - t_start)/60}min.")

    t_end = time.time()
    print(f"Selected methods completed. Total time: {(t_end - t_start)/60}min.")
