# Customization

This covers the three ways you can put your own stamp on Soupy: editing the prompts (personality, search voice, image-prompt rewriting, etc.), changing the web panel's colors, and rebranding the panel itself.

## Personality and prompts

Soupy's personality is one system prompt, `behaviour`, sent with every chat reply. Eight more prompts cover the other things Soupy writes. Each is a plain text file in `prompts/`:

| Prompt | Shipped file | Used for |
|--------|--------------|----------|
| Chat personality | `behaviour.default.txt` | The system prompt for every chat reply |
| Search answers | `behaviour_search.default.txt` | `/soupysearch` summaries |
| Article posts | `behaviour_daily_post.default.txt` | Commentary on the daily article post |
| 9-ball | `nineball.default.txt` | `/9ball` answers |
| Fancy rewrite | `fancy.default.txt`, `fancy_flux.default.txt` | Rewriting an image prompt for the Fancy button (Stable Diffusion, Flux) |
| Random prompt | `randomprompt.default.txt`, `randomprompt_flux.default.txt` | Turning random keywords into an image prompt (Stable Diffusion, Flux) |
| Negative prompt | `sd_negative_prompt.default.txt` | Sent with every image request |

The bot loads `prompts/<name>.txt` when that file exists and the shipped `prompts/<name>.default.txt` otherwise. The `.txt` files are yours: git ignores them, so an update never overwrites one.

### Editing a prompt

The panel's **Model & Personality** tab has an editor for all nine. It shows each file exactly as stored and saves it exactly as typed. A save writes `prompts/<name>.txt` and keeps the version it replaced in `prompts/backups/` (the newest 20 per prompt). "Use shipped default" moves your file into the backups and returns to the shipped text.

The files can also be edited in any text editor. Copy `prompts/behaviour.default.txt` to `prompts/behaviour.txt` and edit the copy.

The bot reads prompts once, when it starts. Restart it after a change ("Save + restart bot" in the editor does both).

### Prompts left in `.env-stable`

Older installs kept the prompts in `.env-stable` (`BEHAVIOUR`, `BEHAVIOUR_SEARCH`, `BEHAVIOUR_DAILY_POST`, `9BALL`, `FANCY`, `RANDOMPROMPT`, `SD_NEGATIVE_PROMPT`). A value still set there overrides the prompt file, and the panel shows it read-only. To move them into files:

```bash
python tools/migrate_prompts.py
```

It writes each one to `prompts/<name>.txt` and comments out the line in `.env-stable`.

## Web Control Panel Customization

The Web Control Panel category in the Environment Editor exposes every theming variable with a color-picker UI. Or edit `.env-stable` directly:

### Color scheme

```bash
WEB_COLOR_PAGE_BG=#1e1010
WEB_COLOR_CARD_BG=#1a2332
WEB_COLOR_TEXT_PRIMARY=#e5e7eb
# ... 30+ more WEB_COLOR_* variables in .env-stable.example
```

The full set covers page background, cards, text (primary/secondary/muted), tabs (active/inactive backgrounds, text, borders), tab content, console drawer, status indicators, env editor field colors, and env editor popovers. There's a complete listing in `.env-stable.example` and the Environment Editor's Web Control Panel tab.

### Title and binding

```bash
WEB_CONTROL_PANEL_TITLE="My Custom Bot Control"
SOUPY_WEB_HOST=0.0.0.0
SOUPY_WEB_PORT=4941
```

`WEB_CONTROL_PANEL_TITLE` rebrands the browser tab title and the heading. `SOUPY_WEB_HOST` and `SOUPY_WEB_PORT` change where the panel listens — useful if you have something else on `4941` or want to bind to `127.0.0.1` only.
