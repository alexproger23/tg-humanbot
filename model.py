import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from torch.utils.data import IterableDataset, DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time


class FineTunnedGPT:
    def __init__(self, model: str, load=False, tokenizer_path=r"save_models/tokenizer_model.json"):
        self.train_loss = []
        self.trainable_parameters = []

        if load:
            assert model is not None, "error, none path"
            assert tokenizer_path is not None, "error, none tokenizer path"

            self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", load_in_8bit=True)
            self.load_tokenizer(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            config = AutoConfig.from_pretrained(model)
            config_dict = config.to_dict()
            if "loss_type" in config_dict:
                del config_dict["loss_type"]
            new_config = config.__class__(**config_dict)
            self.model = GPT2LMHeadModel.from_pretrained(model, config=new_config)

    def load_tokenizer(self, save_path=r"save_models/tokenizer_model.json"):
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

    def freeze_model_layers(self, heads, separate=False):
        if separate and self.model.lm_head.weight.data_ptr() == self.model.transformer.wte.weight.data_ptr():
            # выходное преобразование эмбеддингов в токены отделяется от входного, чтобы можно было отдельно его обучать
            self.model.lm_head = nn.Linear(
                self.model.config.n_embd,
                self.model.config.vocab_size,
                bias=False
            )

            self.model.lm_head.weight.data = torch.clone(self.model.transformer.wte.weight.data)

            assert self.model.lm_head.weight.data_ptr() != self.model.transformer.wte.weight.data_ptr(), \
                "Error, lm_head layer concat with wte layer"

            for param in self.model.transformer.lm_head.parameters():
                param.requires_grad = True
                self.trainable_parameters.append(param)

        for name, param in self.model.named_parameters():
            if "wte" in name:
                param.requires_grad = True
                self.trainable_parameters.append(param)
            else:
                param.requires_grad = False

        if heads:
            for head in heads:
                for name, param in self.model.named_parameters():
                    if "transformer.h." + str(head) + "." in name:
                        param.requires_grad = True
                        self.trainable_parameters.append(param)

    def train_tokenizer(self, path):
        os.makedirs("save_models", exist_ok=True)
        save_path = r"save_models/tokenizer_model.json"

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

        self.load_tokenizer(save_path)

    def train(self,
              dataset,
              epoch=5,
              lr=0.002,
              force_train=False,
              heads=None,
              batch_size=4,
              device="cpu",
              save_path=r"save_models/finetune_gpt2"):

        assert torch.device(device) == torch.device("cuda" if torch.cuda.is_available() else "cpu"), "Error, wrong device"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if device == "cuda":
            scaler = GradScaler()

        if force_train or not os.path.exists(save_path):

            if heads is None:
                heads = []
            self.freeze_model_layers(heads=heads)

            self.model = self.model.to(device)
            self.model.train()

            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=2,
                                    pin_memory=True)
            optimizer = optim.Adam(self.trainable_parameters, lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=0.0001)

            epoch_progress_bar = tqdm(range(epoch), desc="Epoch", dynamic_ncols=True)
            #iter_progress_bar = tqdm(dataloader, desc="Iter", total=len(dataloader), leave=False, dynamic_ncols=True)

            for ep in epoch_progress_bar:
                start_epoch = start_time = time.time()
                for iteration, batch in enumerate(dataloader):
                    ids, mask, labels = batch
                    ids = ids.to(device)
                    mask = mask.to(device)
                    labels = labels.to(device)
                    labels[mask == 0] = -100

                    with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                        output = self.model(input_ids=ids, attention_mask=mask, labels=labels, past_key_values=None)
                        loss = output.loss

                    if device == "cuda":
                        scaler.scale(loss).backward()

                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    self.train_loss.append(loss.item())
                    if iteration % 100 == 0:
                        self.model.save_pretrained(save_path, safe_serialization=True)
                        time_100 = (time.time() - start_time) / 60
                        time_all = (time.time() - start_epoch) / 60
                        remained_time = (len(dataloader) // 100) * time_100 - time_all
                        print(f"Iteration: {iteration}/{len(dataloader)}, Loss: {loss.item()}, time~{remained_time}")
                        start_time = time.time()

                self.model.save_pretrained(save_path, safe_serialization=True)
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
                top_p=0.8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                temperature=0.9
            )
        outputs = outputs.tolist()[0][max_prompt_len:]
        print(outputs)
        answer = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return answer