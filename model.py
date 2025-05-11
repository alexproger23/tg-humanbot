import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from torch.utils.data import IterableDataset, DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm
import os

from dataset import TrainDataset

class FineTunnedGPT:
    def __init__(self, model: str):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        config = AutoConfig.from_pretrained(model)
        config_dict = config.to_dict()
        if "loss_type" in config_dict:
            del config_dict["loss_type"]
        new_config = config.__class__(**config_dict)
        self.model = GPT2LMHeadModel.from_pretrained(model, config=new_config)

        self.trainable_parameters = []
        self.freeze_model_layers()

        self.train_loss = []


    def freeze_model_layers(self, unfreeze="lm_head"):
        heads = [11]
        if unfreeze == "lm_head" and self.model.lm_head.weight.data_ptr() == self.model.transformer.wte.weight.data_ptr():
            # выходное преобразование эмбеддингов в токены отделяется от входного, чтобы можно было отдельно его обучать
            self.model.lm_head = nn.Linear(
                self.model.config.n_embd,
                self.model.config.vocab_size,
                bias=False
            )

            self.model.lm_head.weight.data = torch.clone(self.model.transformer.wte.weight.data)

            assert self.model.lm_head.weight.data_ptr() != self.model.transformer.wte.weight.data_ptr(), \
                "Error, lm_head layer concat with wte layer"

        for name, param in self.model.named_parameters():
            #bool_head = sum(("h." + str(head)) in name for head in heads)
            if unfreeze in name:# or bool_head:
                param.requires_grad = True
                self.trainable_parameters.append(param)
            else:
                param.requires_grad = False


    def train_tokenizer(self, path):
        save_path = "save_models\\tokenizer_model.json"
        if not os.path.exists(save_path):
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            special_tokens = ["<unk>"] + ["<user2>", "<user1>", "<eos>"]

            trainer = trainers.BpeTrainer(
                vocab_size=self.model.config.vocab_size,
                min_frequency=2,
                special_tokens=special_tokens
            )

            def file_iterator(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line.strip()

            self.tokenizer.train_from_iterator(file_iterator(path), trainer=trainer)

            self.tokenizer.save(save_path)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=save_path)

        self.tokenizer.model_max_length = 1024
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.eos_token = "<eos>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.additional_special_tokens = ["<user1>", "<user2>"]

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def train(self,
              dataset,
              epoch=5,
              lr=0.001,
              force_train=False,
              device="cuda"):
        save_path = "save_models\\finetune_gpt2"
        if force_train or not os.path.exists(save_path):
            self.model = self.model.to(device)
            self.model.train()

            dataloader = DataLoader(dataset, batch_size=2)
            optimizer = optim.Adam(self.trainable_parameters, lr=lr)

            progress_bar = tqdm(range(epoch), desc="Epoch", dynamic_ncols=True)
            for ep in progress_bar:
                for ids, mask, labels in tqdm(dataloader, desc="Iter", total=len(dataloader)):
                    ids = ids.to(device)
                    mask = mask.to(device)
                    labels = labels.to(device)
                    output = self.model(input_ids=ids, attention_mask=mask, labels=labels, past_key_values=None)
                    loss = output.loss
                    loss.backward()
                    self.train_loss.append(loss.item())

                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.set_postfix(loss=self.train_loss[-1])

            self.model.save_pretrained(save_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(save_path)

    def inference(self, prompt, max_prompt_len=128):
        prompt = "<user1>" + prompt.lower() + "<user1>"
        encoded = self.tokenizer(prompt, padding="max_length", max_length=max_prompt_len, truncation=True,
                                 return_tensors="pt")

        assert self.tokenizer.eos_token_id == self.model.config.eos_token_id, "error, eos token"
        assert max_prompt_len <= 512, "error, max prompt len too long"

        with torch.no_grad():
            outputs = self.model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],  # важно передать attention_mask если есть паддинг!
                max_length=max_prompt_len*2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                num_beams=4,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                temperature=0.9
            )
        outputs = outputs.tolist()[0][max_prompt_len:]
        answer = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return answer