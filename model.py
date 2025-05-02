from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from torch import nn
from datasets import load_dataset


class FineTunnedGPT:
    def __init__(self, model: str):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.model = GPT2LMHeadModel.from_pretrained(model)

        self.freeze_model_layers()

    def freeze_model_layers(self, unfreeze="lm_head"):
        if unfreeze == "lm_head" and self.model.lm_head.weight.data_ptr() == self.model.transformer.wte.weight.data_ptr():
            # выходное преобразование эмбеддингов в токены отделяется от входного, чтобы можно было отдельно его обучать
            self.model.transformer.lm_head = nn.Linear(
                self.model.config.n_embd,
                self.model.config.vocab_size,
                bias=False
            )

            self.model.lm_head.weight.data = self.model.transformer.wte.weight.data.clone()

            assert self.model.lm_head.weight.data_ptr() == self.model.transformer.wte.weight.data_ptr(), \
                "Error, lm_head layer concat with wte layer"

        for name, param in self.model.named_parameters():
            if unfreeze in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def train_tokenizer(self, path, special=["</s>"]):
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["<UNK>"] + special

        trainer = trainers.BpeTrainer(
            vocab_size=10000,
            min_frequency=3,
            special_tokens=special_tokens
        )

        def file_iterator(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()

        self.tokenizer.train_from_iterator(file_iterator(path), trainer=trainer)
        self.tokenizer.save("save_models\\tokenizer_model.json")


    def train(self, path):      # доделать обучени итерационный датасет, токенизирует перед склеиванием в батч, обучение ручками
        self.model.train()

        dataset = load_dataset("text", data_files="path/to/large_file.txt", streaming=True)

        for example in dataset["train"]:
            text = example["text"]
            # Токенизация на лету
            inputs = self.tokenizer(text, truncation=True, max_length=128)
            yield inputs  # или обработка батчами

