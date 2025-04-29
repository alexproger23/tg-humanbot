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

    try:
        pass
        #messages = list(client.iter_messages(user_entity, reverse=True))
    except Exception as e:
        print("User not found")
        return None

    return user_entity

if __name__ == "__main__":
    tgclient = TelegramClient(*get_env())
    tgclient.start()

    messages = []
    target_users = ["Руслан"]

    for target_user in target_users:
        messages.append(get_dialog(tgclient, target_user))

    with open("chat_history.txt", "w", encoding="utf-8") as txt_file:
        for message in tqdm(tgclient.iter_messages(messages[0], reverse=True)):
            print(message)
            if message.text and not message.fwd_from:  # фильтруем пустые или медиа-сообщения
                if "'''" not in message.text and "https://" not in message.text:
                    # записывать в два файла 1. сплошной текст 2. размеченный на 1 сообщение 2 сообщение
                    txt_file.write(f"[{message.date}] {message.sender_id}: {message.text}\n")

    print("Сохранено в chat_history.txt")