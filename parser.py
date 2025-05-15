from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
from telethon.tl.types import User, Dialog

import csv
import dotenv
import os
from tqdm import tqdm

def get_env():
    dotenv.load_dotenv("tg-api.env")
    api_id = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")
    phone = os.getenv("NUMBER")

    return phone, api_id, api_hash

def get_dialog(client, target):
    user_entity = None
    for dialog in client.iter_dialogs():
        if isinstance(dialog.entity, User) and target == dialog.name.strip():
            user_entity = dialog.entity
            break

    if user_entity is None:
        print(f"User {target} not found")
        return None

    return user_entity

if __name__ == "__main__":
    os.makedirs("train_data", exist_ok=True)

    tgclient = TelegramClient(*get_env())
    tgclient.start()

    messages = []
    target_users = ["Руслан"]

    for target_user in target_users:
        entity = get_dialog(tgclient, target_user)
        if entity is not None:
            messages.append(entity)

    i = 0
    with (open("train_data/solid_chat.txt", "w", encoding="utf-8") as solidtext,
          open("train_data/split_chat.txt", "w", encoding="utf-8") as qa_text):
        for chat in messages:
            last_sender = None
            f = 1
            for message in tqdm(tgclient.iter_messages(chat, reverse=True)):
                if message.text and not message.fwd_from:  # фильтруем пустые или медиа-сообщения
                    if "```" not in message.text and "https://" not in message.text:
                        if last_sender is None:
                            qa_text.write("<user1>")
                        elif message.sender_id != last_sender:
                            solidtext.write("\n")
                            if f:
                                qa_text.write("<user1>\n<user2>")
                                f = 0
                            else:
                                qa_text.write("<user2>\n<user1>")
                                f = 1
                        else:
                            solidtext.write(" ")
                            qa_text.write(" ")

                        solidtext.write(message.text.lower())
                        qa_text.write(message.text.lower())
                        last_sender = message.sender_id

            if f:
                qa_text.write("<user1>")
            else:
                qa_text.write("<user2>")