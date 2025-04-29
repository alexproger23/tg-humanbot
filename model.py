from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn


class FineTunnedGPT:
    def __init__(self, model: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
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
