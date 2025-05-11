import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Отключает предупреждение
from datasets import load_dataset

from model import FineTunnedGPT
from dataset import TrainDataset

if __name__ == "__main__":
    # dataset = load_dataset("Den4ikAI/russian_dialogues", streaming=True)
    # print(next(iter(dataset["train"])))

    ftgpt = FineTunnedGPT("ai-forever/rugpt3large_based_on_gpt2")
    ftgpt.train_tokenizer("tg-parser/train_data/solid_chat.txt")

    dataset = TrainDataset(["tg-parser/train_data/solid_chat.txt"], ftgpt.tokenizer)
    ftgpt.train(dataset, 3, force_train=True, device="cuda")

    prompt = "бро"
    answer = ftgpt.inference(prompt)
    print(answer)

    prompt = "Сколько у тебя за пробник?"
    answer = ftgpt.inference(prompt)
    print(answer)

    prompt = "привет бро, как дела"
    answer = ftgpt.inference(prompt)
    print(answer)