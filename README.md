[Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) to help support this project.

# MAJOR UPDATE 9/15/2024

Soupy-solr.py now has all of the below functionality, but has been enhanced in various ways.

1. You can set up a local [Flux Server](https://github.com/black-forest-labs/flux) for image generation tasks.  Use soupy-gradio.py for this function.  This can be used with the ```!flux``` commands, for example ```!flux a picture of a dog``` will generate a picture of a dog and send it to the channel.  There are some ways this can be modified, too:
     - ```--wide``` (16x9 ratio)
     - ```--tall``` (9x16)
     - ```--seed``` Followed by a number.  This way, you can use the same seed (e.g., 12345) and just modify the prompt.
     - ```--n``` Followed by a number, up to 4.  This will create up to 4 variants of a single prompt.
2. Chat history retrieval has been improved.  Soupy's responses will now be more relevant to the current discussion.
3. There are a few places in the script where it says ```/scriptlocation/```.  You must replace this with the location where your script runs.
4. Additionally, I recommend that you update the .env with this pre-prompt to improve Soupy's responses.  This goes in the BEHAVIOUR variable.

```
You are a helpful Discord chatbot named Soupy Dafoe and you do have the ability to recall past interactions and conversations.  Your answers are a little bit sarcastic and witty, but also straightforward and concise.  You give human-like answers rather than lists as your responses.  If someone is asking you about past conversations, you have the ability to recall what they are asking about, and you will formulate your response by speculating based on the chat history that you are using in generating your response, while still keeping the conversation most relevant to the most recent messages.  Be conversational.  Try not to be too random.  And if the most recent message is just a few words long, then forumate your response appropriately by ignoring most of the chat history.  Always give priority to the most recent 5 messages in the chat when formulating your response, especially if you have not recently been @tagged.  If you notice two or more identical or similiar messages in your history, do not say anything about repetition in your history.  Try to be in the moment.
```
An image generated by flux with default settings:

![cat-1](https://github.com/user-attachments/assets/398596e3-4994-4b5f-bc4f-2c0980267c1e)

An image generated by flux with wide setting:

![image](https://github.com/user-attachments/assets/6eeea254-a695-459a-8358-4ac94dcb3b07)

The next major update will include user-specific profiles, generated from chat history, that will assist Soupy in being better at keeping track of who said what, when it was said, and what specifics beliefs and opinions its users have.

# MAJOR UPDATE 6/9/2024, new version is soupy-solr.py

Soupy (soupy-solr.py) now has all of the following functions:

1. Soupy has a rudimentary memory in the form of JSON databases. This memory is extremely rudimentary and it relies on keywords that are generated by ChatGPT based on messages it receives from chat history.  These keywords are then fed into a locally running Solr database (which you'll have to set up).  Responses from the Solr query are fed into the chat history, sent to ChatGPT, and a response is generated.
2. Chat with soupy by @tagging it in any channel that it can access.
3. Specify a specific channel where soupy responds to all messages.
4. !generate an image based on user input.
5. !transform an image based on user input.
6. Describe the image attached to a message.
7. !time <cityname> will return the time in that city.  For example, "!time Belgium" will return the time in Belgium.

This is a major update for soupy.  The new updates are contained in soupy-solr.py.  In a nutshell, all of the below functionality works, plus some new stuff that you'll want to get set up.  There are many small updates, but the main update is as follows:

1. Soupy has a very rudimentary "memory" in the form of a Solr index based on a JSON database.
2. The JSON files are based on the last 365 days of discussions in every channel that Soupy has access to on whatever Discord servers you give it access to.  These JSON files will be stored in whatever directory soupy-solr.py is stored.
3. Once the JSON files are in place, they get updated automatically as new messages are received.
4. The "memory" is extremely rudimentary.  
5. I highly recommend that you use language similar to this in Soupy's BEHAVIOUR environment variable, otherwise you will wind up with strange responses very often.  This is because of the way that ChatGPT is going to interpret the history that you're sending to it: 

```
"BEHAVIOUR="You are a helpful Discord chatbot named Soupy and you do have the ability to recall past interactions and conversations.  Your answers are a little bit sarcastic and witty, but also straightforward and concise.  You give human-like answers rather than lists as your responses.  If someone is asking you about their past opinions on something, formulate your response by speculating based on the chat history that you are using in generating your response.  Be conversational.  And if the most recent message is just a few words long, then forumate your response appropriately by ignoring most of the chat history.  Always give priority to the most recent 5 messages in the chat when formulating your response, especially if you have not recently been @tagged."
```

## How can I make this work??

In order to get soupy-solr.py up and running, you must first install Solr.  I'll leave that up to you to figure out.  It's not that hard, I hope.

Once Solr is installed, create a new core with ```solr create -c soupy```

Once installed, Solr must be properly configured with certain fields.  

```
<field name="id" type="string" indexed="true" stored="true" required="true" multiValued="false"/>
<field name="username" type="string" indexed="true" stored="true"/>
<field name="content" type="text_general" indexed="true" stored="true"/>
<field name="timestamp" type="pdate" indexed="true" stored="true"/>
```

Once all this is complete, it works pretty darn consistently.  It is likely that soupy-solr.py won't run at all without solr properly in place.

## Historical updates below this line.

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







