/*
 * Media tab: the image archive as a thumbnail grid that opens into a lightbox,
 * and the text activity log (media/messages.jsonl) underneath.
 */
(function () {
  const D = window.SoupyDash;
  const e = D.e;
  const { useCallback, useEffect, useRef, useState } = React;

  const PAGE_SIZE = 60;
  const LOG_PAGE_SIZE = 40;

  function thumbUrl(filename) {
    return "/media/thumbs/" + encodeURIComponent(filename || "");
  }

  function imageUrl(filename) {
    return "/media/images/" + encodeURIComponent(filename || "");
  }

  /** Labels for an archive row, keyed off the `source` the API normalizes. */
  function sourceLabel(row) {
    switch ((row && row.source) || "") {
      case "vision":
        return { badge: "V", badgeClass: "media-badge--vision", pill: "Vision", title: "An image Soupy looked at" };
      case "flux":
        return { badge: "Flux", badgeClass: "media-badge--flux", pill: "Flux", title: "Flux (local mflux, text-to-image)" };
      case "flux-img2img":
        return { badge: "Flux i2i", badgeClass: "media-badge--flux", pill: "Flux i2i", title: "Flux (noise-mix image-to-image)" };
      case "flux-edit":
        return {
          badge: "Flux ✎",
          badgeClass: "media-badge--flux-edit",
          pill: "Flux Edit",
          title: "Flux Klein-Edit (reference-image conditioning)",
        };
      default:
        // Rows written before the `source` tag existed came from SD, the only engine then.
        return { badge: "SD", badgeClass: "media-badge--sd", pill: "Stable Diffusion", title: "Stable Diffusion" };
    }
  }

  function Lightbox({ item, position, total, onClose, onStep }) {
    // Each of these remembers which file it is about. A flag reset in an effect
    // loses the race with a cached image, whose load event comes first.
    const [loadedFile, setLoadedFile] = useState("");
    const [missingFile, setMissingFile] = useState("");
    const [copiedFile, setCopiedFile] = useState("");
    const filename = item ? item.filename : "";
    var loaded = !!filename && loadedFile === filename;
    var missing = !!filename && missingFile === filename;
    var copied = !!filename && copiedFile === filename;

    var hasPrev = position != null && position > 0;
    var hasNext = position != null && position < total - 1;
    var vision = !!item && item.event_type === "vision";
    var label = sourceLabel(item);

    function onKey(ev) {
      if (ev.key === "ArrowLeft" && hasPrev) {
        ev.preventDefault();
        onStep(-1);
      } else if (ev.key === "ArrowRight" && hasNext) {
        ev.preventDefault();
        onStep(1);
      }
    }

    function closeOnBackdrop(ev) {
      if (ev.target === ev.currentTarget) onClose();
    }

    var facts = [];
    if (item) {
      if (item.width && item.height) facts.push(["Size", item.width + " × " + item.height]);
      if (!vision && item.seed) facts.push(["Seed", String(item.seed)]);
      // Recorded for Flux pictures since the archive started keeping them.
      if (item.model) {
        var run = item.model;
        if (item.steps != null) run += ", " + D.plural(item.steps, "step");
        if (item.guidance != null) run += ", guidance " + item.guidance;
        facts.push(["Model", run]);
      }
      if (item.strength != null) facts.push(["Strength", String(item.strength)]);
      if (item.seconds != null) facts.push(["Took", item.seconds + " s"]);
      if (item.source_image) {
        facts.push([
          "Made from",
          e(
            "a",
            { href: "/media/" + item.source_image.split("/").map(encodeURIComponent).join("/"), target: "_blank", rel: "noopener" },
            item.source_image.split("/").pop()
          ),
        ]);
      }
      facts.push(["File", item.filename]);
    }

    return e(
      D.Overlay,
      { className: "lightbox", label: "Image viewer", onClose: onClose, onKey: onKey },
      e(
        "div",
        { className: "lightbox-stage", onMouseDown: closeOnBackdrop },
        hasPrev
          ? e(
              "button",
              {
                type: "button",
                className: "lightbox-nav lightbox-nav--prev",
                "aria-label": "Previous image",
                title: "Previous (←)",
                onClick: function () {
                  onStep(-1);
                },
              },
              "‹"
            )
          : null,
        item
          ? e(
              "div",
              { className: "lightbox-frame", onMouseDown: closeOnBackdrop },
              // Until the full file is in, the thumbnail stands in for it, blurred.
              // If the full file is gone, the thumbnail is all there is.
              loaded
                ? null
                : e("img", {
                    key: "preview",
                    className: "lightbox-img" + (missing ? "" : " lightbox-img--preview"),
                    alt: missing ? item.text || "" : "",
                    src: thumbUrl(filename),
                  }),
              missing
                ? null
                : e("img", {
                    key: filename,
                    className: "lightbox-img" + (loaded ? "" : " lightbox-img--loading"),
                    alt: item.text || "",
                    src: imageUrl(filename),
                    onLoad: function () {
                      setLoadedFile(filename);
                    },
                    onError: function () {
                      setMissingFile(filename);
                    },
                  })
            )
          : e("div", { className: "lightbox-frame" }, D.PanelState({ kind: "loading", message: "Loading…" })),
        hasNext
          ? e(
              "button",
              {
                type: "button",
                className: "lightbox-nav lightbox-nav--next",
                "aria-label": "Next image",
                title: "Next (→)",
                onClick: function () {
                  onStep(1);
                },
              },
              "›"
            )
          : null
      ),
      e(
        "aside",
        { className: "lightbox-info" },
        e(
          "div",
          { className: "lightbox-info-head" },
          item ? e("span", { className: "media-badge media-badge--inline " + label.badgeClass, title: label.title }, label.pill) : null,
          position != null
            ? e("span", { className: "lightbox-count" }, D.fmtNum(position + 1) + " of " + D.fmtNum(total))
            : null,
          e(
            "button",
            { type: "button", className: "lightbox-close", "aria-label": "Close", title: "Close (Esc)", onClick: onClose },
            "×"
          )
        ),
        item
          ? e(
              "div",
              { className: "lightbox-info-body" },
              e(
                "div",
                { className: "lightbox-who" },
                vision ? "Posted by " : "Requested by ",
                e("strong", null, item.username || "unknown"),
                item.channel_name ? e("span", { className: "muted" }, " in #" + item.channel_name) : null
              ),
              e("div", { className: "lightbox-when muted" }, D.fmtDateTimeFull(item.ts)),
              e("h6", { className: "lightbox-label" }, vision ? "What Soupy saw" : "Prompt"),
              e(
                "div",
                { className: "lightbox-text" },
                item.text || e("span", { className: "muted" }, vision ? "(no description recorded)" : "(no prompt recorded)")
              ),
              // The prompt as typed, before Fancy rewrote it.
              item.original_prompt ? e("h6", { className: "lightbox-label" }, "Rewritten from") : null,
              item.original_prompt ? e("div", { className: "lightbox-text" }, item.original_prompt) : null,
              missing
                ? e("p", { className: "lightbox-note" }, "The full-size file is gone from media/images; this is the thumbnail.")
                : null,
              e(
                "dl",
                { className: "lightbox-facts" },
                facts.map(function (pair) {
                  return [
                    e("dt", { key: pair[0] + "-t" }, pair[0]),
                    e("dd", { key: pair[0] + "-d", className: pair[0] === "File" ? "mono" : null }, pair[1]),
                  ];
                })
              ),
              e(
                "div",
                { className: "lightbox-actions" },
                item.text
                  ? e(
                      "button",
                      {
                        type: "button",
                        className: "secondary",
                        onClick: async function () {
                          if (await D.copyText(item.text)) setCopiedFile(filename);
                        },
                      },
                      copied ? "Copied" : vision ? "Copy description" : "Copy prompt"
                    )
                  : null,
                missing
                  ? null
                  : e(
                      "a",
                      { role: "button", className: "secondary", href: imageUrl(filename), target: "_blank", rel: "noreferrer" },
                      "Open full size"
                    ),
                item.original_url
                  ? e(
                      "a",
                      {
                        role: "button",
                        className: "secondary",
                        href: item.original_url,
                        target: "_blank",
                        rel: "noreferrer",
                        title: "The attachment on Discord's CDN. These links expire.",
                      },
                      "Discord original"
                    )
                  : null
              )
            )
          : null
      )
    );
  }

  function ActivityLog({ onOpenImage }) {
    const [open, setOpen] = useState(false);
    const [page, setPage] = useState({ items: [], total: 0, offset: 0 });
    const [status, setStatus] = useState("idle");

    const load = useCallback(async function (offset) {
      setStatus("loading");
      try {
        var data = await D.readJson("/api/archive/messages?limit=" + LOG_PAGE_SIZE + "&offset=" + offset);
        setPage({ items: data.items || [], total: Number(data.total || 0), offset: offset });
        setStatus("ready");
      } catch (_err) {
        setStatus("error");
      }
    }, []);

    useEffect(
      function () {
        if (open) load(0);
      },
      [open, load]
    );

    return e(
      "details",
      {
        className: "media-log",
        onToggle: function (ev) {
          setOpen(!!ev.target.open);
        },
      },
      e("summary", null, "Activity log"),
      !open
        ? null
        : e(
            "div",
            { className: "media-log-body" },
            e(
              "div",
              { className: "media-log-head" },
              e("span", { className: "muted" }, "What the bot logged to media/messages.jsonl, newest first."),
              e(D.Pager, { offset: page.offset, limit: LOG_PAGE_SIZE, total: page.total, onPage: load, disabled: status === "loading" })
            ),
            status === "error" ? D.PanelState({ kind: "error", message: "Could not read the activity log." }) : null,
            status === "ready" && !page.items.length ? D.PanelState({ kind: "empty", message: "Nothing logged yet." }) : null,
            e(
              "div",
              { className: "media-log-list" },
              page.items.map(function (row, i) {
                return e(
                  "div",
                  { key: page.offset + "-" + i, className: "media-log-row" },
                  e("span", { className: "media-log-when mono" }, D.fmtDateTime(row.ts)),
                  e("span", { className: "media-log-who" }, row.username || "—"),
                  e("span", { className: "pill" }, row.event_type || "message"),
                  row.image_filename
                    ? e(
                        "a",
                        {
                          href: imageUrl(row.image_filename),
                          className: "media-log-image",
                          onClick: function (ev) {
                            ev.preventDefault();
                            onOpenImage(row);
                          },
                        },
                        "view image"
                      )
                    : null,
                  e(
                    "div",
                    { className: "media-log-text" },
                    D.previewText(row.content, 280) || e("span", { className: "muted" }, "(empty)")
                  )
                );
              })
            )
          )
    );
  }

  function MediaTab({ active }) {
    const [kind, setKind] = useState("all");
    const [query, setQuery] = useState("");
    const search = D.useDebounced(query, 300);
    const [page, setPage] = useState({ items: [], total: 0, offset: 0, counts: null });
    const [status, setStatus] = useState("loading");
    const [error, setError] = useState("");
    // Lightbox: `openIndex` is a place in the filtered archive; `single` is one
    // image opened from the activity log, outside the grid's order.
    const [openIndex, setOpenIndex] = useState(null);
    const [single, setSingle] = useState(null);
    const [lastSeen, setLastSeen] = useState("");
    const ticket = useRef(0);
    const thumbRefs = useRef({});

    const load = useCallback(
      async function (offset) {
        var mine = ++ticket.current;
        setStatus("loading");
        try {
          var qs = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(offset) });
          if (kind !== "all") qs.set("kind", kind);
          if (search.trim()) qs.set("q", search.trim());
          var data = await D.readJson("/api/archive/images?" + qs.toString());
          if (mine !== ticket.current) return;
          setPage({
            items: data.items || [],
            total: Number(data.total || 0),
            offset: Number(data.offset || 0),
            counts: data.counts || null,
          });
          setStatus("ready");
        } catch (err) {
          if (mine !== ticket.current) return;
          setError(err.message);
          setStatus("error");
        }
      },
      [kind, search]
    );

    // `load` changes only when a filter does, and that starts over at the first
    // page. Coming back to the tab reloads the page that was open.
    const lastLoad = useRef(load);
    const offsetRef = useRef(0);
    offsetRef.current = page.offset;
    useEffect(
      function () {
        if (!active) return;
        var sameFilter = lastLoad.current === load;
        lastLoad.current = load;
        load(sameFilter ? offsetRef.current : 0);
      },
      [active, load]
    );

    var current = single || (openIndex != null ? page.items[openIndex - page.offset] || null : null);
    var lightboxOpen = !!single || openIndex != null;

    useEffect(
      function () {
        if (current && !single) setLastSeen(current.filename);
      },
      [current, single]
    );

    // Back in the grid, land on the thumbnail of the last image viewed.
    useEffect(
      function () {
        if (lightboxOpen || !lastSeen) return;
        var el = thumbRefs.current[lastSeen];
        if (el) {
          el.scrollIntoView({ block: "nearest" });
          el.focus({ preventScroll: true });
        }
      },
      [lightboxOpen, lastSeen]
    );

    // Have the neighbours ready before the arrow key asks for them.
    useEffect(
      function () {
        if (openIndex == null) return;
        [openIndex - 1, openIndex + 1].forEach(function (i) {
          var row = page.items[i - page.offset];
          if (row) new Image().src = imageUrl(row.filename);
        });
      },
      [openIndex, page]
    );

    function showAt(index) {
      if (index < 0 || index >= page.total) return;
      setOpenIndex(index);
      // Stepping past the end of the page turns the grid's page too.
      if (index < page.offset || index >= page.offset + page.items.length) {
        load(Math.floor(index / PAGE_SIZE) * PAGE_SIZE);
      }
    }

    function closeLightbox() {
      setOpenIndex(null);
      setSingle(null);
    }

    async function openFromLog(row) {
      var fallback = {
        filename: row.image_filename,
        username: row.username,
        ts: row.ts,
        text: String(row.content || ""),
        event_type: row.event_type === "vision" ? "vision" : "generated",
        source: row.event_type === "vision" ? "vision" : "",
      };
      try {
        var data = await D.readJson("/api/archive/images?limit=5&q=" + encodeURIComponent(row.image_filename));
        var hit = (data.items || []).filter(function (it) {
          return it.filename === row.image_filename;
        })[0];
        setSingle(hit || fallback);
      } catch (_err) {
        setSingle(fallback);
      }
    }

    if (!active) return null;

    var counts = page.counts || {};
    var filtered = kind !== "all" || !!search.trim();

    return e(
      "div",
      { className: "media-tab" },
      e(
        "article",
        { className: "media-gallery" },
        e(
          "div",
          { className: "media-toolbar" },
          e(D.Segmented, {
            label: "Kind of image",
            value: kind,
            onChange: setKind,
            options: [
              { value: "all", label: "All", count: counts.all },
              { value: "vision", label: "Seen by Soupy", count: counts.vision },
              { value: "generated", label: "Generated", count: counts.generated },
            ],
          }),
          e("input", {
            type: "search",
            className: "media-search",
            placeholder: "Search descriptions, prompts, people…",
            "aria-label": "Search images",
            value: query,
            onChange: function (ev) {
              setQuery(ev.target.value);
            },
          }),
          e(D.Pager, { offset: page.offset, limit: PAGE_SIZE, total: page.total, onPage: load, disabled: status === "loading" })
        ),
        status === "error" ? D.PanelState({ kind: "error", message: "Could not load images: " + error }) : null,
        status === "ready" && !page.items.length
          ? D.PanelState({
              kind: "empty",
              message: filtered ? "No images match." : "No images yet. Generated and analyzed images show up here.",
            })
          : null,
        e(
          "div",
          { className: "media-grid" + (status === "loading" ? " media-grid--loading" : "") },
          page.items.map(function (row, i) {
            var label = sourceLabel(row);
            var open = function () {
              showAt(page.offset + i);
            };
            return e(
              "div",
              {
                key: row.filename,
                ref: function (el) {
                  if (el) thumbRefs.current[row.filename] = el;
                  else delete thumbRefs.current[row.filename];
                },
                className: "dash-plain media-thumb" + (row.filename === lastSeen ? " media-thumb--last" : ""),
                role: "button",
                tabIndex: 0,
                title: (row.username || "unknown") + " · " + D.fmtDateTime(row.ts),
                "aria-label": label.pill + " image from " + (row.username || "unknown"),
                onClick: open,
                onKeyDown: function (ev) {
                  if (ev.key === "Enter" || ev.key === " ") {
                    ev.preventDefault();
                    open();
                  }
                },
              },
              e("img", {
                className: "media-thumb-img",
                alt: "",
                loading: "lazy",
                draggable: false,
                src: thumbUrl(row.filename),
                onError: function (ev) {
                  // No thumbnail on disk: fall back to the full image, once.
                  if (ev.target.dataset.fallback) return;
                  ev.target.dataset.fallback = "1";
                  ev.target.src = imageUrl(row.filename);
                },
              }),
              e("span", { className: "media-badge " + label.badgeClass, title: label.title }, label.badge)
            );
          })
        ),
        page.total > PAGE_SIZE
          ? e(
              "div",
              { className: "media-toolbar media-toolbar--foot" },
              e(D.Pager, { offset: page.offset, limit: PAGE_SIZE, total: page.total, onPage: load, disabled: status === "loading" })
            )
          : null
      ),
      e(ActivityLog, { onOpenImage: openFromLog }),
      lightboxOpen
        ? e(Lightbox, {
            item: current,
            position: single ? null : openIndex,
            total: page.total,
            onClose: closeLightbox,
            onStep: function (delta) {
              showAt(openIndex + delta);
            },
          })
        : null
    );
  }

  D.MediaTab = MediaTab;
})();
