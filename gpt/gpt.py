from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import torch
import os


def GPT_inference(model, tokenizer, dataloader, file_path, max_new_len=100, beam_search=1):
    if not os.path.exists(file_path):
        mode = "w"
    else:
        mode = "a"

    model.eval()
    with open(file_path, mode, encoding="utf-8") as f:
        for batch in tqdm(dataloader):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                    max_new_tokens=max_new_len,
                    num_return_sequences=beam_search,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )

            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(generated_texts)
            f.write("\n".join(generated_texts) + "\n")

        f.write("-" * 80 + '\n')