![Soupy Header](https://i.imgur.com/mDrXgrG.png)

Please feel free to [Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) to help support this project.  

# Soupy
Soupy is a chatbot for Discord that can generate images with a local image generator (Flux) and/or with DALL-E 3.  For chatting, it uses a combination of JSONs, ChatGPT, and a local search engine to engage in conversation with its users.  It will index your user's chat messages, and use those messages to create profiles of users.  It will also index every channel on your server to which it has access.  

---
### IMPORTANT - READ THIS, OR ELSE!!
There are multiple versions of soupy.
1. soupy-flux.py: This version of soupy is ONLY the Flux image generation functionality.  It requires soupy-gradio.py to be run simultaneously.
2. soupy-solr.py: This version features user profiles, requires Solr installation and setup, has chat history logging, and rich interactive chatting.  It also includes Flux image generations, and OpenAI/DALL-E 3 image generation.  It requires soupy-gradio.py to be run simultaneously.
3. soupy-classic.py: This version is only chat and DALL-E 3 image generation.  It does not require Solr and does not create user profiles.

Soupy requires OpenAI API access to the ChatGPT models.  Therefore, the chat portion of Soupy uses *real money*.  The DALL-E 3 image generation does, too.  You can skip DALL-E 3 generation and only use Flux locally.

The initial setup, wherein the channel history from your server will be downloaded and indexed and *all of the users on your server will have profiles made of them* costs money via ChatGPT's API.  Some day I will also support local LLMs, but not yet.

To get Flux working, I strongly suggest you start [here, with the official Flux repository](https://github.com/black-forest-labs/flux).  But once you have Flux up-and-running, you can use `soupy-gradio.py`, included in this repository.

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





