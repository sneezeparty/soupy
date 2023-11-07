## Soupy 

### Soupy Duo is a very simple OpenAI/Dall-E-3 API Discord bot.  It's simple to set up and it's simple to run.

1. Install dependencies.
2. create a .env and populate it with your OPENAI_API_KEY and DISCORD_BOT_TOKEN
3. Have fun at $0.04 per image, yikes.

### Soupy is a simple-to-setup/simple-to-use ChatGPT Discord bot that utilizes the ChatGPT API.

The current functionality of the bot is:

1. Soupy will respond to all @mentions of itself in all channels that it can access, *and* to all messages in a specific channel, even if it isn't mentioned.
2. Soupy will also respond to 2% of all messages in all channels that it has access to -- but it will limit its total tokens when responding randomly to MAX_TOKENS_RANDOM.
3. Soupy considers the message history of the channel where it is activated, which by default is the 10 previous messages.
4. Soupy bot will now split any message longer than 1500 characters into multiple chunks, and then send those chunks as separate messages.


#### Set environment variables in .env, which currently includes:

1. Discord bot's token
2. OpenAI API Key
3. Bot's behaviour (attitude, preferences, etc)
4. The number of messages to consider during the bot's response.
5. The max number of tokens to use.
6. A channel for the bot to respond in.
7. The specific model you want to use (GPT-4, gpt-3.5-turbo, etc)







