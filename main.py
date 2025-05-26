from fastapi import FastAPI
from pydantic import BaseModel

from model import FineTunnedGPT

app = FastAPI()
model = FineTunnedGPT(r"save_models/finetune_gpt2",
                      load=True,
                      tokenizer_path=r"save_models/tokenizer_model.json")

class Message(BaseModel):
    text: str

@app.post("/gpt")
async def gpt_response(msg: Message):
    answer = model.inference(msg.text)
    full = answer[0]["generated_text"]
    reply = full[len(msg.text):].strip()
    return {"reply": answer}