import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Отключает предупреждение
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#from datasets import load_dataset

from model import FineTunnedGPT
from dataset import TrainDataset

if __name__ == "__main__":
    # dataset = load_dataset("Den4ikAI/russian_dialogues", streaming=True)
    # print(next(iter(dataset["train"])))

    ftgpt = FineTunnedGPT("ai-forever/rugpt3small_based_on_gpt2")
    ftgpt.train_tokenizer("train_data/solid_chat.txt")

    dataset = TrainDataset(["train_data/solid_chat.txt"], ftgpt.tokenizer)

    ftgpt.train(dataset, 2, heads=[10, 11], force_train=True, device="cuda", save_path=r'drive/MyDrive/modelsaving/coursework')

    print("Ответы: ")
    prompt = "бро"
    answer = ftgpt.inference(prompt)
    print(answer)

    prompt = "Сколько у тебя за пробник?"
    answer = ftgpt.inference(prompt)
    print(answer)

    prompt = "привет бро, как дела"
    answer = ftgpt.inference(prompt)
    print(answer)