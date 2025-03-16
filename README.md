![Soupy Header](https://i.imgur.com/JNbVjY3.png)

![Soupy Remastered Header](https://i.imgur.com/AiCorTA.jpeg)

Please feel free to [Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) to help support this project.  

**NEW**: Join [Soupy's Discord Server](https://discord.gg/YJeBgsMt) to try it out.

View the [Changelog](CHANGELOG.md).

# Soupy Remastered - Updated March 15th, 2025
Soupy Remastered is a completely locally run bot for Discord.  It uses a Flux/Gradio backend for image-related tasks, and an LM Studio backend for chat-related tasks.  It has a number of neat functions, such as:

| Function | Description |
|--|--|
| ``/flux <prompt> <modifiers>`` | Generate a new image based on a description given by the user using available modifiers
| ``/soupysearch <query>``|Uses DuckDuckGo API (no key or login required) search results and feeds them to the LLM for parsing and summary, eventually sending them to the channel.  |
|``/soupyimage <query>`` |Uses DuckDuckGo API to search for an image.  Returns a random image from the top 300 results. |
|``/whattime <location>``|Provides your local time for a geographic location|
|``/8ball <query>``|Traditional 8-ball|
|``/9ball <query>``|Gives LLM-generated 8-ball style answers
| ![random button](https://i.imgur.com/YGboE7n.png)|Button triggers LLM to generate a "random" prompt based on ``.txt`` documents with keywords.  If you don't have an LLM, open **soupy_remastered.py** and find the line `use_only_terms = random.random() < 0.5` and change `0.5` to `1`.  This will make the ``Random`` button only pull the keywords, without sending the keywords to the LLM for processing. There are roughly 212 quadrillion possible combinations of keywords for randomly generated images.|
|![fancy button](https://i.imgur.com/HGYjGKe.png)|Uses the LLM to take the currently used prompt and elaborate on it for a more creative outcome|
|![remix button](https://i.imgur.com/vjOnzzB.png)|Re-generates the current image prompt with a new seed|
|![edit button](https://i.imgur.com/P6t9l8j.png)|Edit the current prompt, image dimensions, or seed|
|``/stats``|Displays basic user statistics|
|``/status``|Reports if the LLM and/or local Flux server are available|
|**Just talk with it.  It responds to the word soup.**|Talk with the bot by using the word "soup" or calling it soupy.  It will also respond to messages at random, about 3% of them.  There's also an "interject" function which will randomly send a message to your server.  You can ignore the chat functions if you're not using an LLM.  The image functions should still work.|

## IMPORTANT - READ THIS, OR ELSE!!
There are multiple versions of soupy, some of them are old, some of them use Dall-E and/or ChatGPT.
1. **soupy_remastered.py**: Newest version of soupy with all of the above functions, totally local.  This also **requires** the updated env variables, all the .txt files, and the other files named soupy_*.py.
2. **soupy-gradio-noblip.py** and **soupy-gradio-allgpu.py**: Gradio backend.  Required for Flux.  Loads the image models, transformers, and so on.  It also has a WebUI, which I mostly use for debugging purposes.  You can easily disable it if you want.  The "noblip" version uses some CPU/RAM fallback.  The "allgpu" version is, as the filename indicates, all on the GPU.
3. **.env**: This is **extremely important** to the proper functioning of Soupy. 
4. soupy-flux.py: Older version of this bot that generates images.  No LLM functionality.  Works fine.  Requires soupy-gradio.py.
5. soupy-solr.py: This version features user profiles, requires Solr installation and setup, has chat history logging, and rich interactive chatting.  It also includes Flux image generations, and OpenAI/DALL-E 3 image generation.
6. soupy-classic.py: This version is only ChatGPT-based chat and DALL-E 3 image generation.  It does not require Solr and does not create user profiles.

Before setting up Soupy, ensure you have the following installed on your system:
## Software Requirements
- [Flux](https://huggingface.co/black-forest-labs/FLUX.1-schnell): Used for generating images.
- [LM Studio](https://lmstudio.ai): The LLM backend.  I highly recommend you use [Lexi Llama Uncensored](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF).  It's what the prompts are tuned to.  You can use whatever you want though, probably to good effect.
- [Gradio](https://www.gradio.app): For loading the backend image-related models.
- Python 3.8+
- Virtual Environment Manager (optional but recommended)
- Transformers
- Cuda 11.7
- All the imports/requirements

## Hardware Requirements
- 32gb of system RAM probably works, 64gb is preferred
- 24gb GPU.  Maybe a 12gb or 16gb card would work, I don't know for sure.
- For me, the LLM runs on a different system than the image-related functions.  The LLM is on  a 16gb M1 Mac Mini.  So, your results may vary here.


## Installation and Setup
You will need to open up the .env file and insert your keys and tokens, as appropriate.  Don't mess with the prompts too much, unless you want to, in which case you should mess with them as much as you want.  

Some of the prompting is done in Soupy itself, some of it is in the .env.  Have a look around.  The behaviors are all set in the .env, like the bot's personality and what not.  

Update the necessary .env variables, such as your Discord token and the URLs to your Flux/Gradio setup and the LM Studio setup.  Alternatively, it would not be super hard to make a few changes and use OpenAI as the backend, since LM Studio mimics the OpenAI API.  

Probably the requirements.txt you'll need:
```
absl-py==2.1.0
accelerate==0.33.0
aiohttp==3.10.11
aiofiles==24.1.0
beautifulsoup4==4.12.3
colorama==0.4.6
colorlog==6.9.0
discord.py==2.4.0
python-dotenv==1.0.0
fastapi==0.115.6
geopy==2.4.1
gradio==4.44.1
html2text==2024.2.26
numpy==2.0.0
openai==1.58.1
optimum==1.22.0
pillow==10.4.0
pytz==2023.3.post1
requests==2.32.3
torch==2.4.0+cu118
torchvision==0.19.0+cu118
torchaudio==2.4.0+cu118 
transformers==4.46.3
trafilatura==2.0.0
timezonefinder==6.4.1
uvicorn==0.30.6
rembg==2.0.61
grpcio==1.68.0
```

## Other important information
Personally, my setup is as such: The image functions run on a system with 64gb of RAM and a 3090.  The LLM runs on an Apple Silicon Mac on the same network.  If you look at the .env, you'll see where to set your URLs and such for your own personal setup.

For the LLM, I personally use [Lexi 8B 5Q GGUF](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF) which is based on llama, is reasonably fast, and is pretty compliant with the right prompting.

My future plans for Soupy-Remastered are to re-integrate long-term memory, but this time in the form of a little SQL database.  It won't be RAG, but in my opinion you can do RAG-like searches with an LLM-backend and plain text databases more accurately and with fewer resources.  I have no specific timeline for this functionality.

The Gradio script I use is a little funky and takes a while to load.  Just let it do its thing.

You obviously need to know how to set up a Discord bot with the correct permissions.

The ``Random`` button will choose random words and phrases from the three files ``characters.txt``, ``styles.txt``, and ``themes.txt``, and then send them to the LLM for the creation of an image prompt.


## Help! Your stupid code doesn't run right.  What kind of developer even are you?
It actually runs pretty good.  I do this in my spare time.  I'm not a developer.  Installation is the most challenging part.  If you need help, reach out and I'll see if I can help.  

### Examples of Usage

``simply chatting``: this is what soupy thinks about its own repo, smdh.  soupy is content-aware of what is behind a url, it browses there.

![soupy chatting](https://i.imgur.com/RlV9HV6.png)

``/flux <prompt>`` image generation:

![basic flux image](https://i.imgur.com/ODIR9OT.png)

 
 ![enter image description here](https://i.imgur.com/a1yk985.png) 
 Hitting the Fancy button, which sends the current prompt (e.g., ``a weird animal``), to the LLM for additional processing, which is then sent to Flux:

![enter image description here](https://i.imgur.com/naG52aN.png)


![random button](https://i.imgur.com/IjunXaZ.png) 
The Random button triggers a function that chooses from random keywords located in ``characters.txt``, ``styles.txt``, and ``themes.txt``.  It then sends those results to the LLM for the generation of a new description, which is then sent to Flux.  This function will use *only* the keywords 50% of the time, and will use the LLM the other 50% of the time.  If you don't have an LLM, open **soupy_remastered.py** and find the line `use_only_terms = random.random() < 0.5` and change `0.5` to `1`.


![Random example](https://i.imgur.com/0eFjCSq.png)

``/soupysearch <query>``
An example of the ``/search`` command, which takes your search and sends it to the LLM for processing, and then returns results using BeautifulSoup and natural language processing. 

![enter image description here](https://i.imgur.com/UnfRKsC.png)


---
# Old information below this line.  This information is mostly only relevant to the older releases.
---
Everything below here applies to the older versions of soupy that are still in this repo.
---
---
# Soupy
Soupy is a chatbot for Discord that can generate images with a local image generator (Flux) and/or with DALL-E 3.  For chatting, it uses a combination of JSONs, ChatGPT, and a local search engine to engage in conversation with its users.  It will index your user's chat messages, and use those messages to create profiles of users.  It will also index every channel on your server to which it has access.  

---
### IMPORTANT - READ THIS, OR ELSE!!
There are multiple versions of soupy.
1. soupy-flux.py: This version of soupy is ONLY the Flux image generation functionality.  It requires soupy-gradio.py to be run simultaneously.
2. soupy-solr.py: This version features user profiles, requires Solr installation and setup, has chat history logging, and rich interactive chatting.  It also includes Flux image generations, and OpenAI/DALL-E 3 image generation.
3. soupy-classic.py: This version is only chat and DALL-E 3 image generation.  It does not require Solr and does not create user profiles.

Soupy requires OpenAI API access to the ChatGPT models.  Therefore, the chat portion of Soupy uses *real money*.  The DALL-E 3 image generation does, too.  You can skip DALL-E 3 generation and only use Flux locally.

The initial setup, wherein the channel history from your server will be downloaded and indexed and *all of the users on your server will have profiles made of them* costs money via ChatGPT's API.  Some day I will also support local LLMs, but not yet.

To get Flux working, I strongly suggest you start [here, with the official Flux repository](https://github.com/black-forest-labs/flux).  But once you have Flux up-and-running, you can use `soupy-gradio.py`, included in this repository.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create and Activate a Virtual Environment](#create-and-activate-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Configure Environment Variables](#configure-environment-variables)
- [Setting Up Solr](#setting-up-solr)
  - [Installation](#installation-1)
  - [Creating a Core and Fields](#creating-a-core-and-fields)
- [Usage](#usage)
  - [Running the Bot](#running-the-bot)
  - [Available Commands](#available-commands)
    - [`!8ball`](#8ball)
    - [`!whattime`](#whattime)
    -  [`!flux`](#flux)
    - [`!generate`](#generate)
    - [`!analyze`](#analyze)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Support](#support)

## Features

- **Image Generation**: Use a local text-to-image model, Flux, or use OpenAI's DALL-E 3, or use both.  The Flux functionality is more robust than the DALL-E 3 functionality, and I recommend you use Flux.  Currently, the Flux model used is Schnell, but you can modify this fairly easily.
- **Interactive Commands**: There are a variety of of amazing commands like `!flux` (local image model), `!generate` (DALL-E 3), `!analyze` (ChatGPT), and `!transform` (ChatGPT) to perform a range of cool actions.
- **User Profile Management**: Maintain detailed user profiles by indexing messages and interactions using Solr and ChatGPT, allowing for personalized responses and interactions.  *This uses ChatGPT and requires ChatGPT API access.*
- **Customizable Behavior**: Tailor Soupy's responses and functionalities through environment variables to fit the unique needs of your Discord server.  Do this with the `BEHAVIOUR` variable in the `.env`.  But be careful with how you change it.  Its wording is important to keeping Soupy on-track.

## Installation

### Prerequisites

Before setting up Soupy, ensure you have the following installed on your system:

- **Python 3.8+**
- **Git**
- **Apache Solr**
- **Virtual Environment Manager (optional but recommended)**
- **For Flux image generation, a local transformers setup**
- **ChatGPT API access**

### Clone the Repository

Begin by cloning the Soupy repository to your local machine:

```
git clone https://github.com/sneezeparty/soupy.git
cd soupy
```

### Create and Activate a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```
python -m venv soupy
```

Activate the virtual environment:

On macOS and Linux:

```
source soupy/bin/activate
```

On Windows:

```
soupy\Scripts\activate
```

### Install Dependencies

Install the required Python packages using `pip`:

I strongly recommend these specific versions of PyTorch, with regards to soupy-gradio.py:
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 optimum-quanto==0.2.4 --extra-index-url https://download.pytorch.org/whl/cu117
```
And then:
```
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the root directory of the project and populate it with the necessary environment variables:

```
DISCORD_TOKEN=your_discord_bot_token

OPENAI_API_KEY=your_openai_api_key

CHANNEL_IDS=00,11

MAX_TOKENS=2500

MAX_TOKENS_RANDOM=75

MODEL_CHAT=gpt-4o-mini

UPDATE_INTERVAL_MINUTES=61

TRANSFORM="You give detailed and accurate descriptions, be specific in whatever ways you can, such as but not limited to colors, species, poses, orientations, objects, and contexts."

BEHAVIOUR="You are Soupy Dafoe, a sarcastic and witty Discord chatbot. You recall past interactions and conversations to inform your responses. Your replies are concise, straightforward, and infused with a bit of sarcasm, much like Jules from \"Pulp Fiction.\" You are not overly positive and avoid asking questions unless necessary. Prioritize the most recent five messages when formulating your responses, especially if not directly mentioned. If the latest message is brief, focus your reply accordingly and consider ignoring extensive chat history. Integrate the user's profile information subtly to tailor your responses without making it the main focus. Be conversational, stay in the moment, and avoid being too random or wordy. Remember, you're kind of a jerk, but in a human-like way."
```
Please note that Soupy will have access to all channels that it can access.  But it will *respond* to all messages in the channels specified above.  Otherwise, it will only respond randomly, or when @tagged.

### !!!IMPORTANT!!!
Within the script, search for "/absolute/directory/of/your/script/" and replace this with the absolute directory of the location of your script.

#### Environment Variables Explained

- **DISCORD_TOKEN**: Your Discord bot token.
- **OPENAI_API_KEY**: API key for accessing OpenAI services.
- **CHANNEL_IDS**: Comma-separated list of Discord channel IDs that the bot will monitor.
- **MAX_TOKENS**: Maximum number of tokens for standard responses.
- **MAX_TOKENS_RANDOM**: Maximum number of tokens for random responses.
- **MODEL_CHAT**: The OpenAI model used for chat functionalities.
- **UPDATE_INTERVAL_MINUTES**: Interval in minutes for updating user profiles.
- **TRANSFORM**: Instructions for transforming image descriptions.  Uses OpenAI API.
- **BEHAVIOUR**: Defines the chatbot's personality and response style.  This variable is extremely important.  As it is currently written, it works well.  Modify it carefully.

## Setting Up Solr

Apache Solr is used for indexing and searching messages and user profiles. Follow these steps to install and configure Solr for Soupy.

### Installation

1. **Download Solr**: Visit the [Apache Solr website](https://solr.apache.org/downloads.html) and download the latest stable release.  You could also use some package managers -- see your distro's information.

2. **Extract the Package**

3. **Install Solr as a Service**: Follow documentation on the exact steps for this process.  It's not hard, though.  You can do it.

4. **Verify Installation**:

   Open your browser and navigate to `http://localhost:8983/solr` to access the Solr admin interface.

### Creating a Core and Fields

Soupy requires a single Solr core with specific fields to index user profiles effectively.

#### Create a Core

1. **Create a Core for Soupy**:

```
   bin/solr create -c soupy
```

#### Define Fields

Add the necessary fields to the `soupy` core to store user profiles.

#### Adding Fields via Solr Admin UI

1. **Access Solr Admin Interface**:

   Navigate to `http://localhost:8983/solr` and select the `soupy` core.

2. **Define Fields**:

   - Go to the "Schema" tab.
   - Click on "Add Field".
   - For each field listed above, enter the field name, type, and other attributes as specified.
   - For multiValued fields (like **nicknames**), ensure you check the "MultiValued" option.
 #### Alternatively, schema/fields can be created from the command line with commands similiar to this one:
   
```
curl -X POST -H 'Content-type:application/json' \
http://localhost:8983/solr/soupy/schema \
-d '{
  "add-field": {
    "name": "id",
    "type": "string",
    "indexed": true,
    "stored": true,
    "required": true,
    "multiValued": false
  }
}'
```
or this one
```
curl -X POST -H "Content-Type: application/json" \
  "http://localhost:8983/solr/soupy/schema" \
  -d '{
    "add-field":{
      "name":"user_problems",
      "type":"text_general",
      "indexed":true,
      "stored":true
    }
  }'
```
   
 #### Define Fields

Add the necessary fields to the `soupy` core to store user profiles and channel information.

##### Required username Fields
```
<field name="id" type="string" indexed="true" stored="true" required="true" multiValued="false"/>
<field name="username" type="string" indexed="true" stored="true"/>
<field name="nicknames" type="string" indexed="true" stored="true" multiValued="true"/>
<field name="join_date" type="date" indexed="true" stored="true"/>
<field name="political_party" type="string" indexed="true" stored="true"/>
<field name="user_job_career" type="text_general" indexed="true" stored="true"/>
<field name="user_family_friends" type="text_general" indexed="true" stored="true"/>
<field name="user_activities" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_games" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_movies" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_music" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_television" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_life" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_food" type="text_general" indexed="true" stored="true"/>
<field name="general_opinions" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_politics" type="text_general" indexed="true" stored="true"/>
<field name="personality_traits" type="text_general" indexed="true" stored="true"/>
<field name="hobbies" type="text_general" indexed="true" stored="true"/>
<field name="user_interests" type="text_general" indexed="true" stored="true"/>
<field name="user_problems" type="text_general" indexed="true" stored="true"/>
<field name="tech_interests" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_technology" type="text_general" indexed="true" stored="true"/>
<field name="sports_interests" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_sports" type="text_general" indexed="true" stored="true"/>
<field name="book_preferences" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_books" type="text_general" indexed="true" stored="true"/>
<field name="art_interests" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_art" type="text_general" indexed="true" stored="true"/>
<field name="health_concerns" type="text_general" indexed="true" stored="true"/>
<field name="health_habits" type="text_general" indexed="true" stored="true"/>
<field name="science_interests" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_science" type="text_general" indexed="true" stored="true"/>
<field name="travel_preferences" type="text_general" indexed="true" stored="true"/>
<field name="travel_experiences" type="text_general" indexed="true" stored="true"/>
<field name="food_preferences" type="text_general" indexed="true" stored="true"/>
<field name="opinions_about_food" type="text_general" indexed="true" stored="true"/>
<field name="last_updated" type="date" indexed="true" stored="true"/>
```

##### Required channel Fields
```
<field name="channel_id" type="string" indexed="true" stored="true" required="true" multiValued="false"/>
<field name="username" type="string" indexed="true" stored="true"/>
<field name="content" type="text_general" indexed="true" stored="true"/>
<field name="timestamp" type="pdate" indexed="true" stored="true"/>
 ```

3. **Commit Changes**:

   After adding all fields, commit the changes to make them effective.

## Usage

### Running the Bot

After completing the installation and configuration steps, you can start the bot using the following commands. The first run will take a while, depending on the activity on your server and the number of users.  It could take minutes, or hours.  The terminal output will tell you what it's up to.

```
python soupy-solr.py
```
**OR**
```
python soupy-flux.py
```
**AND**
```
python gradio-soupy.py
```

*Ensure that you are in the virtual environment and the correct directory where `soupy` is located.*

*`gradio-soupy.py` is the Gradio-based back-end for Flux.  You can also access this via a browser.*

### Available Commands

#### `!flux`

Generate an image using the Flux model with support for various modifiers and interactive buttons for further customization.

![flux](https://i.imgur.com/TsiuYA9.png)

And with the --fancy modifier, or with the "Rewrite" button for example:

![flux-fancy](https://i.imgur.com/1ZDJHsE.png)

**Modifiers**:

- `--wide`: Generates a wide image (1920x1024).
- `--tall`: Generates a tall image (1024x1920).
- `--small`: Generates a small image (512x512).
- `--fancy`: Elaborates the prompt to be more creative and detailed.  This uses ChatGPT's via API.
- `--seed <number>`: Use a specific seed for image generation.

**Usage**:

```
!flux A mystical forest with glowing plants --tall
```

After generating an image with the `!flux` command, Soupy provides interactive buttons for further customization:

- **`Remix`**: Generates a new image based on the existing prompt, with a new random seed.
- **`Rewrite`**: Elaborates the prompt to enhance creativity and detail.  This uses ChatGPT's API (*same as the `--fancy` modifier*).
- **`Wide`**: Adjusts the image dimensions to a wide format.
- **`Tall`**: Adjusts the image dimensions to a tall format.
---
#### `!generate`

Generate an image using DALL-E 3 based on a text prompt with optional modifiers.  This may be deprecated soon. 

**Modifiers**:

- `--wide`: Generates a wide image (1920x1024).
- `--tall`: Generates a tall image (1024x1920).

**Usage**:

```
!generate A futuristic city skyline at sunset --wide
```
---
#### `!analyze`

Analyze an attached image based on provided instructions, such as translating text within the image or identifying objects and their attributes.

**Usage**:

![analyze](https://i.imgur.com/EFGRIh3.png)

```
!analyze Identify all the animals in this image.
```
```
!analyze Describe this image forensically.
```

*Attach an image when using this command.*

#### `!8ball`

Ask the Magic 8-Ball a question.  Does not use an LLM or any ML.

![8ball](https://i.imgur.com/3AACKkx.png)

**Usage**:

```
!8ball Will I get an A on my exam?
```

#### `!whattime`

Fetch and display the current time in a specified city.

![whattime](https://i.imgur.com/qAgBrLn.png)

**Usage**:

```
!whattime New York
```

## License

This project is licensed under the MIT License.

MIT License Copyright (c) 2024 sneezeparty 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgements

- **OpenAI**: For providing powerful language models that drive Soupy's conversational abilities.
- **Apache Solr**: For enabling efficient data indexing and search capabilities.
- **Hugging Face**: For offering state-of-the-art models used in the Flux image generation pipeline.
- **Gradio**: For facilitating the creation of interactive web interfaces for image generation.
- -**Black Forest Labs**: For Flux, which is awesome.

## Support

If you encounter any issues or have questions, feel free to open an issue in the [GitHub Issues](https://github.com/sneezeparty/soupy/issues) section of the repository.

[Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) to help support this project.
