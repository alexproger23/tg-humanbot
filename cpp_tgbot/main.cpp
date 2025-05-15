#include <tgbot/tgbot.h>
#include <iostream>

int main() {
    TgBot::Bot bot("YOUR_TELEGRAM_BOT_TOKEN");

    bot.getEvents().onCommand("start", [&bot](TgBot::Message::Ptr message) {
        bot.getApi().sendMessage(message->chat->id, "Привет! Я бот.");
        });

    try {
        std::cout << "Бот запущен..." << std::endl;
        bot.start();
    }
    catch (std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }

    return 0;
}
