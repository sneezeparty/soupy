"""Which links in a message got no working preview, and the fixer link to post instead.

Instagram and Facebook give Discord nothing to show (no preview at all, or a
"log in" card), and the others fail now and then. Fixer sites serve the same
post with a preview Discord can play: ``instagram.com/reel/X`` becomes
``oginstagram.com/reel/X``. Fixer sites die every year or so, which is why the
domain for each site is a setting, ``LINK_FIX_SITES``.

The cog (``soupy.cogs.linkfix``) waits for Discord to build a message's
previews, describes each one as a :class:`Preview`, and asks :func:`fixes`
which links still need one. Nothing here touches Discord or the network.

A link is left alone when it is:

* wrapped in ``<...>`` (the poster turned its preview off), spoilered, or in code;
* not a post (a profile page, an Instagram share link the fixer can't read);
* already on the fixer site we would use (re-posting it would change nothing).

A link on another fixer site counts like the site itself: if it got no preview
(that fixer is dead), it is rewritten to the current one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

logger = logging.getLogger(__name__)

# How a preview has to look before the site's link counts as working.
ANY = "any"  # any preview at all (X, Bluesky and Reddit build text-only ones for text posts)
MEDIA = "media"  # a picture or a video: a bare title card is a login wall
VIDEO = "video"  # a picture or a video, not just a thumbnail: TikTok's thumbnail card doesn't play

MAX_LINKS = 5


@dataclass(frozen=True)
class Site:
    name: str
    hosts: Tuple[str, ...]  # the site's own domains
    mirrors: "re.Pattern[str]"  # fixer domains people already post for this site
    posts: Tuple["re.Pattern[str]", ...]  # matched against "host/path?query"; group 1 is the post id
    wants: str
    default_fixer: str
    keep_params: FrozenSet[str] = frozenset()
    short_subdomains: FrozenSet[str] = frozenset()  # kept on the fixer: vt.tiktok.com -> vt.tnktok.com


SITES: Tuple[Site, ...] = (
    Site(
        "instagram",
        ("instagram.com",),
        re.compile(r"(?:^|\.)(?:[a-z]+instagram|instagram[a-z]+)\.com$"),
        # /share/ links are left alone: the fixer can't read them (404 when tried, Nov 2026).
        (re.compile(r"^[^/]+/(?:(?!share/)[\w.]+/)?(?:p|reels?|tv)/([\w-]+)"),),
        MEDIA,
        "oginstagram.com",
        keep_params=frozenset({"img_index"}),
    ),
    Site(
        "x",
        ("x.com", "twitter.com"),
        re.compile(r"(?:^|\.)(?:fxtwitter|vxtwitter|fixupx|fixvx|twittpr)\.com$"),
        (re.compile(r"^[^/]+/(?:\w+|i(?:/web)?)/status(?:es)?/(\d+)"),),
        ANY,
        "fxtwitter.com",
    ),
    Site(
        "bluesky",
        ("bsky.app",),
        re.compile(r"(?:^|\.)(?:fxbsky|bskx|bskye|bskyx|vxbsky)\.app$"),
        (re.compile(r"^[^/]+/profile/[^/]+/post/([a-z0-9]+)"),),
        ANY,
        "bskx.app",
    ),
    Site(
        "reddit",
        ("reddit.com",),
        re.compile(r"(?:^|\.)(?:rxddit|vxreddit)\.com$"),
        (
            re.compile(r"^[^/]+/(?:r|u|user)/[\w-]+/comments/([a-z0-9]+)"),
            re.compile(r"^[^/]+/r/\w+/s/(\w+)"),
        ),
        ANY,
        "vxreddit.com",
    ),
    Site(
        "tiktok",
        ("tiktok.com",),
        re.compile(r"(?:^|\.)(?:tnktok|kktiktok|vxtiktok|tfxktok|tiktxk)\.com$"),
        (
            re.compile(r"^[^/]+/@[\w.-]+/(?:video|photo)/(\d+)"),
            re.compile(r"^[^/]+/t/(\w+)"),
            re.compile(r"^(?:vt|vm)\.[^/]+/(\w+)"),
        ),
        VIDEO,
        "tnktok.com",
        short_subdomains=frozenset({"vt", "vm"}),
    ),
    Site(
        "facebook",
        ("facebook.com",),
        re.compile(r"(?:^|\.)(?:facebed|fxfb|facebookez)\.com$"),
        (
            re.compile(r"^[^/]+/share/[a-z]/(\w+)"),
            re.compile(r"^[^/]+/(?:[\w.]+/)?(?:reel|videos|posts)/([\w.]+)"),
            re.compile(r"^[^/]+/watch/?\?(?:.*&)?v=(\d+)"),
            re.compile(r"^[^/]+/(?:permalink|story)\.php\?(?:.*&)?story_fbid=(\w+)"),
        ),
        MEDIA,
        "facebed.com",
        keep_params=frozenset({"v", "story_fbid", "id", "fbid"}),
    ),
)
_BY_NAME = {s.name: s for s in SITES}

DEFAULT_SITES = ", ".join(f"{s.name}={s.default_fixer}" for s in SITES)

# Subdomains that mean nothing to the fixer (www.x.com, m.facebook.com, old.reddit.com).
_NOISE_SUBDOMAIN = re.compile(r"^(?:www|m|mobile|old|new|np)\.")

_URL = re.compile(r"<?https?://[^\s<>|()\[\]]+>?", re.I)
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_SPOILER = re.compile(r"\|\|.*?\|\|", re.S)
_EMBED_NOTE = re.compile(r"\[Embed [^\]]*\]?")


@dataclass(frozen=True)
class Preview:
    """What Discord built for one link (a ``discord.Embed``, flattened)."""

    url: str
    kind: str = "rich"  # "components" is the newer layout fixer sites use, with its media inside
    image: bool = False
    thumbnail: bool = False
    video: bool = False


@dataclass(frozen=True)
class Link:
    url: str
    site: Site
    post_id: str
    on_mirror: bool


@lru_cache(maxsize=8)
def parse_sites(raw: str) -> Dict[str, str]:
    """``"instagram=oginstagram.com, x=fxtwitter.com"`` -> ``{"instagram": "oginstagram.com", ...}``.

    Empty means every site with its default fixer. A site left out is not fixed.
    """
    raw = (raw or "").strip() or DEFAULT_SITES
    out: Dict[str, str] = {}
    for part in raw.split(","):
        name, _, fixer = part.partition("=")
        name, fixer = name.strip().lower(), fixer.strip().lower()
        fixer = re.sub(r"^https?://", "", fixer).strip("/")
        if name not in _BY_NAME:
            if name:
                logger.warning("LINK_FIX_SITES: unknown site %r (known: %s)", name, ", ".join(_BY_NAME))
            continue
        if fixer:
            out[name] = fixer
    return out


def _host_path(url: str) -> Optional[Tuple[str, str, str, str]]:
    """``(host, normalized host, path, query)``, or None when it isn't a web address."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    return host, _NOISE_SUBDOMAIN.sub("", host), parts.path or "/", parts.query


def _on(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def site_of(url: str, fixers: Dict[str, str]) -> Optional[Tuple[Site, bool]]:
    """The site a URL is a link to, and whether it is on a fixer site. Disabled sites are None."""
    hp = _host_path(url)
    if hp is None:
        return None
    _host, host, _path, _query = hp
    for site in SITES:
        if site.name not in fixers:
            continue
        if any(_on(host, d) for d in site.hosts):
            return site, False
        if site.mirrors.search(host) or _on(host, fixers[site.name]):
            return site, True
    return None


def post_id(url: str, site: Site) -> Optional[str]:
    hp = _host_path(url)
    if hp is None:
        return None
    _host, host, path, query = hp
    target = f"{host}{path}" + (f"?{query}" if query else "")
    for pattern in site.posts:
        m = pattern.match(target)
        if m:
            return m.group(1)
    return None


def find_links(content: str, fixers: Dict[str, str]) -> List[Link]:
    """Links to posts on an enabled site, in order, without repeats (at most ``MAX_LINKS``)."""
    if not content or "http" not in content:
        return []
    text = _SPOILER.sub(" ", _CODE.sub(" ", content))
    out: List[Link] = []
    seen = set()
    for raw in _URL.findall(text):
        if raw.startswith("<"):
            continue  # <https://...> turns the preview off on purpose
        url = raw.rstrip(">").rstrip(".,!?;:'\"")
        found = site_of(url, fixers)
        if not found:
            continue
        site, on_mirror = found
        pid = post_id(url, site)
        if not pid or (site.name, pid) in seen:
            continue
        seen.add((site.name, pid))
        out.append(Link(url, site, pid, on_mirror))
        if len(out) >= MAX_LINKS:
            break
    return out


def rewrite(link: Link, fixer: str) -> Optional[str]:
    """The same post on the fixer site, without tracking parameters. None when it is already there."""
    hp = _host_path(link.url)
    if hp is None:
        return None
    host, normalized, path, query = hp
    if _on(host, fixer):
        return None
    sub = normalized.split(".", 1)[0]
    new_host = f"{sub}.{fixer}" if sub in link.site.short_subdomains else fixer
    kept = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k in link.site.keep_params]
    return f"https://{new_host}{path}" + (f"?{urlencode(kept)}" if kept else "")


def preview_works(preview: Preview, wants: str) -> bool:
    if wants == ANY or preview.kind == "components":
        return True
    if wants == VIDEO:
        return preview.video or preview.image
    return preview.video or preview.image or preview.thumbnail


def fixes(links: Sequence[Link], previews: Sequence[Preview], fixers: Dict[str, str]) -> List[str]:
    """Fixer links for every link whose post got no working preview."""
    out: List[str] = []
    for link in links:
        mine = [p for p in previews if _same_post(p.url, link, fixers)]
        if not mine and sum(1 for other in links if other.site is link.site) == 1:
            # One link to this site: Discord may have filed its preview under the
            # resolved address (a short link's full one), so any preview from the site is its.
            mine = [p for p in previews if (site_of(p.url, fixers) or (None,))[0] is link.site]
        wants = ANY if link.on_mirror else link.site.wants
        if any(preview_works(p, wants) for p in mine):
            continue
        fixed = rewrite(link, fixers[link.site.name])
        if fixed:
            out.append(fixed)
    return out


def _same_post(url: str, link: Link, fixers: Dict[str, str]) -> bool:
    found = site_of(url, fixers)
    return bool(found) and found[0] is link.site and post_id(url, link.site) == link.post_id


def is_bare_links(text: str) -> bool:
    """True when a message is only links (as Soupy's fix replies are), ignoring the archive's embed notes."""
    tokens = _EMBED_NOTE.sub(" ", text or "").split()
    return bool(tokens) and all(re.match(r"^<?https?://\S+$", t) for t in tokens)
