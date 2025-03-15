## Changelog

## [0.8.4] - 3/15/2025
### Changed or added
- Changed the behavior of the Random button in the following way: 50% of the time it will now choose keywords and send the naked keywords to the Flux backend for generation, rather than using the LLM for elaboration.  The results of this are pretty cool, and in some ways more varied than the LLM-generated responses
- Added a new command, /soupyimage <query>, which uses the DuckDuckGo API to search for an image.  It will choose a random image from the top 300 results and send that image to the channel.
- Removed BLIP functionality because it takes up too much memory and doesn't add much functionality
- Added a new version of the gradio backend that is 100% GPU without and CPU/RAM fallback

### Fixed
- The queueing system for the Random button would sometimes lose items due to how the back-end was functioning
- Small bug fixes

## [0.8.31] - 3/2/2025
### Fixed
- Fancy button was broken during last update, it works again now
- Too many grey sphynx cats were being generated
- Significantly improved /soupysearch functionality by limiting the context size, adding URL validation and ranking system, and just making the whole thing actually work
- Gave soupy the ability to parse URLs in chat in order to integrate the content of those URLs into its chat history during conversation

## [0.8.3] - 3/1/2025
### Fixed
- Redid the queue system so that it will cope better with different types of requests simultaneously.
- Fixed a problem where sometimes the Random button would be ignored.

## [0.8.21] - 1-21-2025
### Fixed
- Completely revamped the search functionality to use DuckDuckGo instead of Google.

## [0.8.2] - 1-17-2025
### Fixed
- Modified the BLIP behavior in the Gradio backend.  It now loads on-demand and unloads when not needed, which speeds up image generation tasks by freeing up system resources.
- Interject was not properly integrating message history, and was not properly using the INTERJECT .env variable.
- Modified /search method to return more relevant results.

## [0.8.1] - 1-13-2025
### Fixed
- Corrected shutdown behavior

### Added
- Split search functions into separate search.py cog.
- Improved search functionality overall
- Improved the bot's ability to track user stats across several servers simultaneously
- Improved queuing behavior
- Created BEHAVIOUR_ALT and SPECIAL_GUILD_ID .env variables to allow the bot to have different "personalities" across different servers, if wanted.
- Renamed all soupy_remastered.py related files to a standard soupy_*.py format

## [0.8.0] - 2025-01-10
- Initial release of the project.
