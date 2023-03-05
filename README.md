##Soupy 
###Soupy is a ChatGPT Discord bot that utilizes the ChatGPT API.

The current functionality of the bot is:

1. Soupy will respond to all @mentions of itself in all channels that it can access, *and* to all messages in a specific channel, even if it isn't mentioned.
2. Soupy considers the message history of the channel where it is activated, which by default is the 8 previous messages.
3. Soupy bot will now split any message longer than 1500 characters into multiple chunks, and then send those chunks as separate messages.

Set environment variables in .env, which currently includes:

1. Discord bot's token
2. OpenAI API Key
3. Bot's behaviour (attitude, preferences, etc)
4. The number of messages to consider during the bot's response.
5. A channel for the bot to respond in.







