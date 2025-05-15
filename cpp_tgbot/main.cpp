#include <tgbot/tgbot.h>
#include <iostream>

int main() {
    TgBot::Bot bot("YOUR_TELEGRAM_BOT_TOKEN");

    bot.getEvents().onCommand("start", [&bot](TgBot::Message::Ptr message) {
        bot.getApi().sendMessage(message->chat->id, "������! � ���.");
        });

    try {
        std::cout << "��� �������..." << std::endl;
        bot.start();
    }
    catch (std::exception& e) {
        std::cerr << "������: " << e.what() << std::endl;
    }

    return 0;
}
