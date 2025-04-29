from torch.utils.data import Dataset
import torch
import os


class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=1024):
        script_dir = os.path.dirname(os.path.normpath(__file__))
        if __name__ == "__main__":
            file_path = os.path.join(script_dir, data_path)
        else:
            file_path = os.path.join(script_dir, "..", data_path)
        self.file_path = os.path.normpath(file_path)

        try:
            self.file_content = open(self.file_path, "r", encoding="utf-8").readlines()
        except FileNotFoundError:
            print("Prompt file not found")
            self.file_content = []

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

            return len(lines)

    def __getitem__(self, idx):
        prompt = self.file_content[idx].strip()
        encoded_input = self.tokenizer(prompt,
                                       padding='do_not_pad',
                                       truncation=True,
                                       max_length=self.max_len)

        return torch.tensor(encoded_input["input_ids"]), torch.tensor(encoded_input["attention_mask"])