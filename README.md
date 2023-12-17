# !! Only soupy-combined is currently functioning correctly

Pretty big update on 12/17.  

Soupy does more stuff now.

1. Chat with soupy by @tagging it in any channel that it can access.
2. Specify a specific channel where soupy responds to all messages.
3. !generate an image based on user input.
4. !transform an image based on user input.
5. Describe the image attached to a message.

A generated image:
![generate](https://github.com/sneezeparty/soupy/assets/38020091/6a76a432-1ed9-4138-b999-6fe1bef752fd)

The instruction to transform the image:
![transform-instruction](https://github.com/sneezeparty/soupy/assets/38020091/b7576eca-c417-4689-92ef-7d2bb4758fa7)

The transformed image:
![transform](https://github.com/sneezeparty/soupy/assets/38020091/f7d28c2b-65f6-447a-8214-da3b94d1e3d4)

An image description:
![description](https://github.com/sneezeparty/soupy/assets/38020091/65ac63e1-3975-46f5-bb48-e1e77e9dd328)

Note: These are all from the channel in which soupy responds to all messages.  That's an environment variable.

To describe an image, simply attach it to a message and @tag the bot.

Pretty big update on 12/10/2023 -- soupy can now analyze images sent to the channel.  It will also follow instructions on how you want the image to be analyzed.  For example, "describe this as a 5 year old would describe it" or "give a poetic description of this image."

To use this functionality, just tag the bot in a message with an attached image, and optionally give an instruction about how you would like the image to be analyzed.

## Soupy -- Requires OpenAI API access.

### Soupy-combined combines the functionality of the chat bot with the functionality of the image generation bot.

Currently, soupy-combined.py uses the chat model specified in the .env, and the image !generate uses a hard-coded model which IS NOT referenced in the .env.

1. Install dependencies.
2. Create a .env and populate it.
3. Have fun.






