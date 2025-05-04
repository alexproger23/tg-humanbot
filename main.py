from model import FineTunnedGPT


if __name__ == "__main__":
    ftgpt = FineTunnedGPT("gpt2")
    ftgpt.train_tokenizer("tg-parser/train_data/solid_chat.txt")

    ftgpt.train(["tg-parser/train_data/solid_chat.txt"], 3, device="cpu")

    print(ftgpt.tokenizer.eos_token, ftgpt.tokenizer.eos_token_id)
    prompt = "бро"
    answer = ftgpt.inference(prompt)
    print(answer)

    prompt = "егэ"
    answer = ftgpt.inference(prompt)
    print(answer)