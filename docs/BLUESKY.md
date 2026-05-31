# Bluesky Integration

Soupy maintains an autonomous Bluesky presence via the AT Protocol, running alongside the Discord bot. Activity is randomized across the day so it doesn't look like a scheduled bot, with a configurable minimum gap between actions (default 45 minutes).

## Autonomous daily activity (6am–11pm Pacific)

- **Replies** — Default 4–7/day. Finds interesting posts from the timeline, trending topics, and thread exploration; reads the full comment thread; generates 3 candidate replies and picks the best.
- **Quote-posts** — Default 1/day. Shares someone else's post with brief commentary.
- **Original posts** — Default 1/day. Mines Bluesky and DuckDuckGo for articles, fetches full content, generates a take, and posts with a link card and og:image thumbnail.
- **Likes** — Likes 1–2 good comments per thread it replies to. Daily cap is now env-tunable via `BLUESKY_MAX_LIKES_PER_DAY` (default 10).
- **Follows** — May follow up to a few interesting accounts per day based on their recent post quality. Daily cap is env-tunable via `BLUESKY_MAX_FOLLOWS_PER_DAY` (default 2).

The Overview tab on the web panel shows today's progress live (e.g. `5 replies · 1 post · 1 repost`) and includes a one-click toggle for the whole loop (`BLUESKY_AUTO_REPLY`).

## Quality controls

- **3-candidate generation** — Every reply, quote-post, and original post generates 3 candidates; an LLM judge picks the best one.
- **Fact-checking** — Posts and replies are checked against the source article before publishing. The checker explicitly allows opinion, framing, sarcasm, and editorial takes — it only blocks invented facts, wrong attribution, and contradictions.
- **Author diversity** — Won't reply to the same author twice in a row (last 5 authors blocked, next 10 penalized).
- **Article freshness** — Articles without a verifiable date are rejected outright; anything older than `BLUESKY_ARTICLE_FRESHNESS_DAYS` (default 14) is filtered out.
- **Schedule persistence** — Daily schedule is saved to disk and survives restarts; completed actions are not re-scheduled.
- **Rate limiting** — Random delays between all API actions; minimum inter-action gap is `BLUESKY_MIN_GAP_MINUTES` (default 45); daily caps on likes and follows.

## Configuration

Editable from the web Environment Editor (Bluesky Integration tab):

- **`BLUESKY_HANDLE`** — Your handle (e.g. `name.bsky.social`).
- **`BLUESKY_APP_PASSWORD`** — App password from Settings → App Passwords (not your main password).
- **`BLUESKY_AUTO_REPLY`** (default `false`) — Master enable/disable for the autonomous loop.
- **`BLUESKY_REPLIES_MIN`** (default `4`) — Lower bound on replies per day.
- **`BLUESKY_REPLIES_MAX`** (default `7`) — Upper bound on replies per day.
- **`BLUESKY_REPOSTS_PER_DAY`** (default `1`) — Quote-posts per day.
- **`BLUESKY_POSTS_PER_DAY`** (default `1`) — Original article posts per day.
- **`BLUESKY_MIN_GAP_MINUTES`** (default `45`) — Minimum gap between any two autonomous actions.
- **`BLUESKY_MAX_LIKES_PER_DAY`** (default `10`) — Daily cap on likes.
- **`BLUESKY_MAX_FOLLOWS_PER_DAY`** (default `2`) — Daily cap on follows.
- **`BLUESKY_ARTICLE_FRESHNESS_DAYS`** (default `14`) — Reject articles older than this when picking originals.

The last four used to be hardcoded in `soupy/cogs/bluesky.py` and were lifted into env vars during M2 work — they are now tunable from the panel without code changes. Defaults match the historical hardcoded values.

## Manual control (`/soupysky`)

- `/soupysky` or `/soupysky reply` — Find and reply to a post now.
- `/soupysky repost` — Quote-post something interesting now.
- `/soupysky post` — Find an article and post about it now.
- `/soupysky post url:https://...` — Post about a specific article you provide.

All actions report results to the configured musing channel and tag the owner. The same logic that drives the autonomous loop also handles `/soupysky` triggers, so what you get manually is what you'd get automatically.
