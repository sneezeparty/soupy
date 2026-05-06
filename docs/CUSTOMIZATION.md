# Customization

This covers the three ways you can put your own stamp on Soupy: editing the prompts (personality, search voice, image-prompt rewriting, etc.), changing the web panel's colors, and rebranding the panel itself.

## Personality (`BEHAVIOUR`)

Soupy's personality is a single system prompt: the `BEHAVIOUR` variable. The `BEHAVIOUR_SEARCH` variable defines the voice used for `/soupysearch` summaries. Both can be edited from the web panel's **Model & Personality** tab, which has a dedicated editor with auto-formatting on load and a "Load raw" mode for the as-stored text — plus a "Save preset + restart" button.

For very long rewrites, editing the underlying prompt file in a real text editor is the safest path; the web editor's textarea can get unwieldy past a few hundred lines.

### Prompt files (since M2)

As of M2, prompts no longer live exclusively in `.env-stable`. They live as plain text files in `prompts/`:

- `prompts/behaviour.default.txt` — main personality used for chat replies, musings, autonomous interjections.
- `prompts/behaviour_search.default.txt` — voice for `/soupysearch`.
- `prompts/behaviour_daily_post.default.txt` — voice used to write daily article posts.
- `prompts/fancy.default.txt` — the Fancy-button image-prompt rewrite template.
- `prompts/randomprompt.default.txt` — the random-keyword-to-prompt template.
- `prompts/nineball.default.txt` — `/9ball` response style.
- `prompts/sd_negative_prompt.default.txt` — default negative prompt for image generation.

To customise, you have two options:

1. **Edit the `.default.txt` file directly.** Quick and obvious. Be aware that a future update may overwrite it.
2. **Copy the default to a non-default override.** Copy `prompts/behaviour.default.txt` to `prompts/behaviour.txt` (drop `.default`) and edit that. The bot prefers the non-default file when present, so updates to the default won't clobber your version.

If you have an existing `.env-stable` with a heavily customised `BEHAVIOUR` string and want to migrate it into the `prompts/` flow, run:

```bash
python tools/migrate_prompts.py
```

It pulls `BEHAVIOUR` / `BEHAVIOUR_SEARCH` / etc. out of `.env-stable` and writes them into the corresponding override files in `prompts/`, so you don't lose any of the work you put into the prompt over time.

> **Don't paste extremely long prompts through the web env editor.** The form-based editor was built around short scalar values; the personality strings can be very long, and the web UI may truncate or escape them in surprising ways. Edit the `prompts/*.txt` files (or `.env-stable` directly) in a real editor for big rewrites.

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
