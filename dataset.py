from torch.utils.data import IterableDataset


class TrainDataset(IterableDataset):
    def __init__(self, pathes, tokenizer):
        self.pathes = pathes
        self.tokenizer = tokenizer

    def __iter__(self):
        for path in self.pathes:
            with open(path, "r", encoding="utf-8") as f:
                while True:
                    line1 = f.readline().strip()
                    line2 = f.readline().strip()

                    if not line1 or not line2:
                        break

                    yield process(line1, line2, self.tokenizer)

    def __len__(self):
        lines = 0
        for path in self.pathes:
            with open(path, "r", encoding="utf-8") as f:
                lines += len(f.readlines())
        return lines // 2

class JsonToTxt:
    def __init__(self, dataset, path, tokenizer):
        self.dataset = dataset
        self.path = path
        self.tokenizer = tokenizer

    def __iter__(self):
        pass


def process(question, answer, tokenizer):
    question = "<user1>" + question + "<user1>"
    text = question + "<user2>" + answer + "<eos>"
    encoded = tokenizer(text.lower(), padding="max_length", max_length=512, truncation=True,
                             return_tensors="pt")
    ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    question_len = len(tokenizer(question.lower(), padding="max_length", max_length=512, truncation=True,
                                      return_tensors="pt"))
    labels = ids.clone()
    labels[:question_len] = -100

    return ids, attention_mask, labels