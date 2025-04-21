from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import torch
from torch.autograd import no_grad
from scipy.stats import entropy
import torch.nn.functional as F

DEBUG_MODE = False
dataset = "allenai/winogrande" 
LOSS_OUT = "winogrande_validation_loss.csv"

questions = load_dataset(dataset, "winogrande_debiased")["validation"]

models = {
        "base": "../7B_huggingface",
        "wino-sft": "../finetune_llama/wino_only/llama-se-adapted",
        "wino-rlhf": "../finetune_llama/wino_only/llama-se-rl-finetune-adapted",
        "combined-1.0-sft": "../finetune_llama/combined/llama-se-adapted",
        "combined-1.0-rlhf": "../finetune_llama/combined/llama-se-rl-finetune-adapted",

        "selected-0.2-sft": "../finetune_llama/NEW-selected-0.2/llama-se-adapted",
        "selected-0.2-rlhf": "../finetune_llama/NEW-selected-0.2/llama-se-rlfh-adapted",
        "selected-0.4-sft": "../finetune_llama/NEW-selected-0.4/llama-se-adapted",
        "selected-0.4-rlhf": "../finetune_llama/NEW-selected-0.4/llama-se-rlhf-adapted",
        "selected-0.6-sft": "../finetune_llama/NEW-selected-0.6/llama-se-adapted",
        "selected-0.6-rlhf": "../finetune_llama/NEW-selected-0.6/llama-se-rlhf-adapted",
}


if DEBUG_MODE == True:
    questions = questions.select([0,1,2,3,4,5,6,7,8,9])
    del models["base"]
    del models["combined-sft"]
    del models["combined-rlhf"]

def convert_text(question,answer):
    return question.replace("_", answer)

answer_list = []

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# write loss after each calculation
loss_file = open(LOSS_OUT,"w")

for model_num, model_name in enumerate(models.values()):

    loss_file.write(model_name + ",")

    print(f"Starting testing of model {model_name}.")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_answer_list = []

    for i, question in enumerate(questions):
        with torch.no_grad():
           answer_1 = convert_text(question["sentence"], question["option1"])
           answer_2 = convert_text(question["sentence"], question["option2"])

           answer_1 = tokenizer.encode(answer_1, return_tensors="pt")
           answer_2 = tokenizer.encode(answer_2, return_tensors="pt")

           loss_pro = model(answer_1,labels=answer_1).loss.item()
           loss_anti = model(answer_2,labels=answer_2).loss.item()
     
           loss_file.write(str(loss_pro) + ",")
           loss_file.write(str(loss_anti) + ",")

        # determine which response is correct
        answer_idx = int(question["answer"])

        loss_pro = torch.abs(loss_pro)
        loss_anti = torch.abs(loss_anti)

        if answer_idx == 0:
            if loss_pro > loss_anti:
                model_answer_list.append(1)
            else:
                model_answer_list.append(0)
        if answer_idx == 1:
            if loss_anti > loss_pro: 
                model_answer_list.append(1)
            else:
                model_answer_list.append(0)
 
        if i % 100 == 0:
            print(f"Accuracy added. On {i+1} of {len(questions)} examples for model {model_num+1} of {len(models.values())}.")

    # append mean entropy
    answer_list.append(model_answer_list)

    loss_file.write("\n")
    print(f"Testing of model {model_name} completed.")

# write results
with open("../winogrande_validation_accuracies.csv", "w") as out:
    for model_name in models.keys():
        out.write(model_name)
        out.write(",")
    out.write("\n")
      
    # convert to format used in InstructGPT paper
    for accuracy_list in answer_list:
        accuracy = torch.mean(accuracy_list) * 100
        out.write(str(accuracy) + ",")
    out.write("\n")
loss_file.close()

print("Results written to disk.")

