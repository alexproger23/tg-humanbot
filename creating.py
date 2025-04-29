import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Отключает предупреждение

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from gpt.gpt import GPT_inference
from gpt.dataset import PromptDataset

def creatingGPTexamples(model, tokenizer):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    inference_dataset = PromptDataset("data\\prompts.txt", tokenizer)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1)

    GPT_inference(model, tokenizer, inference_dataloader, "output/gpt_results.txt", max_new_len=40)


if __name__ == "__main__":

    model = "ai-forever/rugpt3large_based_on_gpt2" # "gpt2"

    creatingGPTexamples(AutoModelForCausalLM.from_pretrained(model), AutoTokenizer.from_pretrained(model))