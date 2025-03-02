## Changelog

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
