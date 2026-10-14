/*
 * Database tab: browse a server's archived messages, read the member profiles
 * built from them (and Soupy's memory of itself), and run the jobs that keep
 * both current. One server at a time; the picker is shared with the Overview tab.
 */
(function () {
  const D = window.SoupyDash;
  const e = D.e;
  const { useEffect, useMemo, useRef, useState } = React;

  const PAGE_SIZES = [25, 50, 100, 200];
  const CONTEXT_SPAN = 8;
  const CONTEXT_SPAN_WIDE = 25;
  // The profile job is polled fast only while it is doing something, so an
  // idle dashboard stays out of the server log.
  const BATCH_POLL_ACTIVE_MS = 12000;
  const BATCH_POLL_IDLE_MS = 120000;
  const BATCH_POLL_ERR_MS = 30000;

  const EMPTY_FILTERS = {
    search: "",
    channel: "",
    member: "",
    memberName: "",
    from: "",
    to: "",
    hasImages: false,
    hasUrls: false,
    order: "desc",
  };

  function guildUrl(prefix, guild) {
    return prefix + encodeURIComponent(guild);
  }

  function displayName(row) {
    return String(row.nickname || "").trim() || row.username || "unknown";
  }

  function stamp(row) {
    return row.date + " " + row.time;
  }

  /** A <input type="date"> value as the local start (or end) of that day. */
  function localDay(value, endOfDay) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value || "");
    if (!m) return null;
    return endOfDay
      ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), 23, 59, 59)
      : new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), 0, 0, 0);
  }

  function activate(handler) {
    return function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        handler();
      }
    };
  }

  /** State that goes back to `initial` when `scope` changes, with no stale render in between. */
  function useScoped(scope, initial) {
    const [box, setBox] = useState({ scope: scope, value: initial });
    var value = box.scope === scope ? box.value : initial;
    return [
      value,
      function (next) {
        setBox({ scope: scope, value: next });
      },
    ];
  }

  /** `text` with every case-insensitive occurrence of `term` wrapped in <mark>. */
  function highlight(text, term) {
    var s = String(text || "");
    var q = String(term || "").trim().toLowerCase();
    if (!q) return s;
    var low = s.toLowerCase();
    var parts = [];
    var from = 0;
    var at;
    while ((at = low.indexOf(q, from)) !== -1) {
      if (at > from) parts.push(s.slice(from, at));
      parts.push(e("mark", { key: at }, s.slice(at, at + q.length)));
      from = at + q.length;
    }
    if (from < s.length) parts.push(s.slice(from));
    return parts;
  }

  // ---------------------------------------------------------------------------
  // Messages
  // ---------------------------------------------------------------------------

  function MessageDetail({ guild, row, onOnlyMember, onOnlyChannel }) {
    // null (closed) | "loading" | "error" | { rows }
    const [context, setContext] = useState(null);
    const [span, setSpan] = useState(CONTEXT_SPAN);

    function loadContext(n) {
      setSpan(n);
      setContext("loading");
      D.readJson(guildUrl("/api/database/context/", guild) + "/" + row.message_id + "?before=" + n + "&after=" + n)
        .then(setContext)
        .catch(function () {
          setContext("error");
        });
    }

    function extra(label, text) {
      return e(
        "div",
        { className: "msg-extra" },
        e("div", { className: "msg-extra-label" }, label),
        e("div", { className: "msg-extra-text" }, D.linkify(text))
      );
    }

    return e(
      "div",
      { className: "msg-detail" },
      e(
        "div",
        { className: "msg-detail-text" },
        row.message_content ? D.linkify(row.message_content) : e("span", { className: "muted" }, "(no text)")
      ),
      row.image_description ? extra("What Soupy saw in the image", row.image_description) : null,
      row.url_summary ? extra("Link summary", row.url_summary) : null,
      e(
        "div",
        { className: "msg-detail-meta mono" },
        "@" + row.username + " · user " + row.user_id + " · message " + row.message_id + " · " + stamp(row) + " UTC"
      ),
      e(
        "div",
        { className: "msg-detail-actions" },
        e(
          "button",
          {
            type: "button",
            className: "secondary",
            onClick: function () {
              if (context) setContext(null);
              else loadContext(CONTEXT_SPAN);
            },
          },
          context ? "Hide conversation" : "Show conversation"
        ),
        e("button", { type: "button", className: "secondary", onClick: onOnlyMember }, "Only " + displayName(row)),
        e("button", { type: "button", className: "secondary", onClick: onOnlyChannel }, "Only #" + row.channel_name)
      ),
      context === "loading" ? D.PanelState({ kind: "loading", message: "Loading the conversation…", inline: true }) : null,
      context === "error" ? D.PanelState({ kind: "error", message: "Could not load the conversation." }) : null,
      context && context.rows
        ? e(
            "div",
            { className: "msg-context" },
            context.rows.map(function (r) {
              return e(
                "div",
                { key: r.message_id, className: "msg-context-row" + (r.message_id === row.message_id ? " msg-context-row--focus" : "") },
                e("span", { className: "msg-context-when", title: stamp(r) + " UTC" }, D.fmtDateTime(stamp(r))),
                e("span", { className: "msg-context-who" }, displayName(r)),
                e(
                  "span",
                  { className: "msg-context-text" },
                  r.message_content ? D.linkify(r.message_content) : e("span", { className: "muted" }, "(no text)")
                )
              );
            }),
            span < CONTEXT_SPAN_WIDE
              ? e(
                  "button",
                  {
                    type: "button",
                    className: "secondary msg-context-more",
                    onClick: function () {
                      loadContext(CONTEXT_SPAN_WIDE);
                    },
                  },
                  "Show more around it"
                )
              : null
          )
        : null
    );
  }

  function MessagesView({ guild, filters, onFilters }) {
    const [limit, setLimit] = useState(50);
    const [offset, setOffset] = useState(0);
    const [page, setPage] = useState({ rows: [], total: 0 });
    const [status, setStatus] = useState("loading");
    const [error, setError] = useState("");
    const [openId, setOpenId] = useState(null);
    const [members, setMembers] = useState([]);
    const [channels, setChannels] = useState([]);
    const search = D.useDebounced(filters.search, 300);
    const ticket = useRef(0);
    const topRef = useRef(null);

    useEffect(
      function () {
        setMembers([]);
        setChannels([]);
        if (!guild) return;
        var cancelled = false;
        D.readJson(guildUrl("/api/database/members/", guild))
          .then(function (data) {
            if (!cancelled) setMembers(data.members || []);
          })
          .catch(function () {});
        D.readJson(guildUrl("/api/channels/", guild))
          .then(function (data) {
            if (!cancelled) setChannels(data.channels || []);
          })
          .catch(function () {});
        return function () {
          cancelled = true;
        };
      },
      [guild]
    );

    // Everything that picks which messages match. A change here goes back to page one.
    var queryKey = [
      guild,
      search.trim(),
      filters.channel,
      filters.member,
      filters.from,
      filters.to,
      filters.hasImages,
      filters.hasUrls,
      filters.order,
      limit,
    ].join("|");
    const lastKey = useRef(queryKey);

    useEffect(
      function () {
        if (!guild) return;
        if (lastKey.current !== queryKey) {
          lastKey.current = queryKey;
          setOpenId(null);
          if (offset !== 0) {
            setOffset(0);
            return;
          }
        }
        var mine = ++ticket.current;
        setStatus("loading");
        var qs = new URLSearchParams({ limit: String(limit), offset: String(offset), order: filters.order });
        if (search.trim()) qs.set("search", search.trim());
        if (filters.channel) qs.set("channel_id", filters.channel);
        if (filters.member) qs.set("user_id", filters.member);
        if (filters.hasImages) qs.set("has_images", "true");
        if (filters.hasUrls) qs.set("has_urls", "true");
        var from = localDay(filters.from, false);
        var to = localDay(filters.to, true);
        if (from) qs.set("after", D.utcStamp(from));
        if (to) qs.set("before", D.utcStamp(to));
        D.readJson(guildUrl("/api/database/explore/", guild) + "?" + qs.toString())
          .then(function (data) {
            if (mine !== ticket.current) return;
            setPage({ rows: data.rows || [], total: Number(data.total || 0) });
            setStatus("ready");
          })
          .catch(function (err) {
            if (mine !== ticket.current) return;
            setError(err.message);
            setStatus("error");
          });
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [queryKey, offset]
    );

    function setFilter(patch) {
      onFilters(Object.assign({}, filters, patch));
    }

    function turnPage(next) {
      setOffset(next);
      setOpenId(null);
      if (topRef.current) topRef.current.scrollIntoView({ block: "nearest" });
    }

    var dirty = Object.keys(EMPTY_FILTERS).some(function (k) {
      return filters[k] !== EMPTY_FILTERS[k];
    });
    var memberListed = members.some(function (m) {
      return m.user_id === filters.member;
    });

    var pager = e(D.Pager, { offset: offset, limit: limit, total: page.total, onPage: turnPage, disabled: status === "loading" });

    return e(
      "div",
      { className: "db-pane", ref: topRef },
      e(
        "div",
        { className: "msg-filters" },
        e("input", {
          type: "search",
          className: "msg-filter-search",
          placeholder: "Search messages, image descriptions, link summaries…",
          "aria-label": "Search messages",
          value: filters.search,
          onChange: function (ev) {
            setFilter({ search: ev.target.value });
          },
        }),
        e(
          "select",
          {
            "aria-label": "Channel",
            value: filters.channel,
            onChange: function (ev) {
              setFilter({ channel: ev.target.value });
            },
          },
          e("option", { value: "" }, "All channels"),
          channels.map(function (ch) {
            return e("option", { key: ch.id, value: ch.id }, "#" + ch.name + " (" + D.fmtNum(ch.messages) + ")");
          })
        ),
        e(
          "select",
          {
            "aria-label": "Member",
            value: filters.member,
            onChange: function (ev) {
              setFilter({ member: ev.target.value, memberName: "" });
            },
          },
          e("option", { value: "" }, "Everyone"),
          // A member picked from a profile may post too little to be in the list.
          filters.member && !memberListed
            ? e("option", { value: filters.member }, filters.memberName || "user " + filters.member)
            : null,
          members.map(function (m) {
            return e("option", { key: m.user_id, value: m.user_id }, m.label + " (" + D.fmtNum(m.message_count) + ")");
          })
        ),
        e(
          "label",
          { className: "msg-filter-date" },
          "From",
          e("input", {
            type: "date",
            value: filters.from,
            max: filters.to || null,
            onChange: function (ev) {
              setFilter({ from: ev.target.value });
            },
          })
        ),
        e(
          "label",
          { className: "msg-filter-date" },
          "To",
          e("input", {
            type: "date",
            value: filters.to,
            min: filters.from || null,
            onChange: function (ev) {
              setFilter({ to: ev.target.value });
            },
          })
        ),
        e(
          "label",
          { className: "msg-filter-check" },
          e("input", {
            type: "checkbox",
            checked: filters.hasImages,
            onChange: function () {
              setFilter({ hasImages: !filters.hasImages });
            },
          }),
          "Has image"
        ),
        e(
          "label",
          { className: "msg-filter-check" },
          e("input", {
            type: "checkbox",
            checked: filters.hasUrls,
            onChange: function () {
              setFilter({ hasUrls: !filters.hasUrls });
            },
          }),
          "Has link"
        ),
        e(
          "select",
          {
            "aria-label": "Order",
            value: filters.order,
            onChange: function (ev) {
              setFilter({ order: ev.target.value });
            },
          },
          e("option", { value: "desc" }, "Newest first"),
          e("option", { value: "asc" }, "Oldest first")
        ),
        dirty
          ? e(
              "button",
              {
                type: "button",
                className: "secondary",
                onClick: function () {
                  onFilters(EMPTY_FILTERS);
                },
              },
              "Clear filters"
            )
          : null
      ),
      e(
        "div",
        { className: "db-pane-bar" },
        pager,
        e(
          "label",
          { className: "db-page-size" },
          "Per page",
          e(
            "select",
            {
              value: String(limit),
              onChange: function (ev) {
                setLimit(Number(ev.target.value || 50));
              },
            },
            PAGE_SIZES.map(function (n) {
              return e("option", { key: n, value: String(n) }, String(n));
            })
          )
        )
      ),
      status === "error" ? D.PanelState({ kind: "error", message: "Could not load messages: " + error }) : null,
      status === "ready" && !page.rows.length
        ? D.PanelState({ kind: "empty", message: dirty ? "No messages match these filters." : "This archive has no messages yet." })
        : null,
      e(
        "div",
        { className: "msg-list" + (status === "loading" ? " msg-list--loading" : "") },
        page.rows.map(function (r) {
          var open = openId === r.message_id;
          var toggle = function () {
            // A click that ends a text selection is a copy, not a toggle.
            if (String(window.getSelection ? window.getSelection() : "")) return;
            setOpenId(open ? null : r.message_id);
          };
          return e(
            "div",
            { key: r.message_id, className: "msg" + (open ? " msg--open" : "") },
            e(
              "div",
              { className: "dash-plain msg-row", role: "button", tabIndex: 0, "aria-expanded": open ? "true" : "false", onClick: toggle, onKeyDown: activate(toggle) },
              e("span", { className: "msg-when", title: stamp(r) + " UTC" }, D.fmtDateTime(stamp(r))),
              e("span", { className: "msg-who" }, displayName(r)),
              e("span", { className: "msg-where" }, "#" + r.channel_name),
              e(
                "span",
                { className: "msg-text" },
                r.message_content ? highlight(r.message_content, search) : e("span", { className: "muted" }, "(no text)"),
                r.image_description ? e("span", { className: "pill msg-flag" }, "image") : null,
                r.url_summary ? e("span", { className: "pill msg-flag" }, "link") : null
              )
            ),
            open
              ? e(MessageDetail, {
                  guild: guild,
                  row: r,
                  onOnlyMember: function () {
                    setFilter({ member: r.user_id, memberName: displayName(r) });
                  },
                  onOnlyChannel: function () {
                    setFilter({ channel: r.channel_id });
                  },
                })
              : null
          );
        })
      ),
      page.rows.length >= 15 ? e("div", { className: "db-pane-bar db-pane-bar--foot" }, pager) : null
    );
  }

  // ---------------------------------------------------------------------------
  // Profiles
  // ---------------------------------------------------------------------------

  function itemWhen(item) {
    var first = D.fmtMonth(item.date);
    var again = D.fmtMonth(item.last_seen);
    if (first && again && again !== first) return first + " – " + again;
    return first;
  }

  function itemWhenTitle(item) {
    if (!item.date) return null;
    return "First came up " + item.date + (item.last_seen ? " · last confirmed " + item.last_seen : "");
  }

  function profileAsText(profile) {
    if (profile.summary) return profile.summary;
    var out = [profile.name, ""];
    if (profile.overview) out.push("Overview:", profile.overview, "");
    if (profile.communication_style) out.push("Communication style:", profile.communication_style, "");
    (profile.sections || []).forEach(function (sec) {
      if (!sec.items.length) return;
      out.push(sec.label + ":");
      sec.items.forEach(function (it) {
        out.push("• " + (it.name ? it.name + ": " : "") + it.text + (it.date ? " (" + it.date + ")" : ""));
      });
      out.push("");
    });
    return out.join("\n").trim();
  }

  function ProfileDetail({ guild, id, term, epoch, onShowMessages }) {
    const [profile, setProfile] = useState(null);
    const [status, setStatus] = useState("loading");
    const [section, setSection] = useState("all");
    const [filter, setFilter] = useState("");
    const [raw, setRaw] = useState(false);
    const [copied, setCopied] = useState(false);

    useEffect(
      function () {
        if (!guild || !id) return;
        var cancelled = false;
        setStatus("loading");
        D.readJson(guildUrl("/api/profiles/get/", guild) + "/" + encodeURIComponent(id))
          .then(function (data) {
            if (cancelled) return;
            setProfile(data.profile);
            setStatus("ready");
          })
          .catch(function () {
            if (!cancelled) setStatus("error");
          });
        return function () {
          cancelled = true;
        };
      },
      [guild, id, epoch]
    );

    useEffect(
      function () {
        setSection("all");
        setRaw(false);
        setCopied(false);
      },
      [id]
    );

    // Searching the member list for a phrase narrows the open profile to it too,
    // unless the search matched this profile by name and nothing in it says the phrase.
    useEffect(
      function () {
        var q = String(term || "").trim().toLowerCase();
        var said =
          !!q &&
          !!profile &&
          profile.user_id === id &&
          (profile.sections || []).some(function (sec) {
            return sec.items.some(function (it) {
              return (it.text + " " + (it.name || "")).toLowerCase().indexOf(q) !== -1;
            });
          });
        setFilter(said ? term.trim() : "");
      },
      [term, id, profile]
    );

    const sections = useMemo(
      function () {
        var q = filter.trim().toLowerCase();
        return ((profile && profile.sections) || []).map(function (sec) {
          var items = sec.items
            .filter(function (it) {
              return !q || it.text.toLowerCase().indexOf(q) !== -1 || String(it.name || "").toLowerCase().indexOf(q) !== -1;
            })
            .slice()
            .sort(function (a, b) {
              var ra = a.last_seen || a.date || "";
              var rb = b.last_seen || b.date || "";
              return ra < rb ? 1 : ra > rb ? -1 : 0;
            });
          return Object.assign({}, sec, { shown: items });
        });
      },
      [profile, filter]
    );

    if (status === "error") return D.PanelState({ kind: "error", message: "Could not load this profile." });
    if (!profile) return D.PanelState({ kind: "loading", message: "Loading profile…" });

    var isSelf = profile.kind === "self";
    var coverage = profile.coverage || {};
    var filled = sections.filter(function (sec) {
      return sec.items.length;
    });
    var empty = sections.filter(function (sec) {
      return !sec.items.length;
    });
    var visible = filled.filter(function (sec) {
      return (section === "all" || section === sec.key) && sec.shown.length;
    });
    var filtering = !!filter.trim();
    var matchCount = filled.reduce(function (n, sec) {
      return n + sec.shown.length;
    }, 0);

    var facts = [];
    facts.push(D.plural(profile.message_count, "message") + " read");
    facts.push(D.plural(profile.item_count, "item"));
    if (coverage.first && coverage.last) facts.push("covers " + D.fmtMonth(coverage.first) + " – " + D.fmtMonth(coverage.last));
    if (profile.updated_at) facts.push("updated " + D.timeAgo(profile.updated_at));
    if (profile.model_used) facts.push(profile.model_used);

    return e(
      "div",
      { className: "profile" + (status === "loading" ? " profile--loading" : "") },
      e(
        "header",
        { className: "profile-head" },
        e(
          "div",
          { className: "profile-head-main" },
          e("h3", { className: "profile-name" }, profile.name),
          isSelf
            ? e("span", { className: "pill profile-kind" }, "Soupy's memory of itself")
            : e("span", { className: "profile-id mono", title: "Discord user ID" }, profile.user_id)
        ),
        e("div", { className: "profile-facts" }, facts.join(" · ")),
        e(
          "div",
          { className: "profile-actions" },
          isSelf
            ? null
            : e(
                "button",
                {
                  type: "button",
                  className: "secondary",
                  onClick: function () {
                    onShowMessages(profile.user_id, profile.name);
                  },
                },
                "Their messages →"
              ),
          e(
            "button",
            {
              type: "button",
              className: "secondary",
              onClick: async function () {
                setCopied(await D.copyText(profileAsText(profile)));
              },
            },
            copied ? "Copied" : "Copy as text"
          ),
          e(
            "button",
            {
              type: "button",
              className: "secondary",
              "aria-pressed": raw ? "true" : "false",
              onClick: function () {
                setRaw(!raw);
              },
            },
            raw ? "Hide raw data" : "Raw data"
          )
        )
      ),
      profile.version === 1
        ? D.PanelState({
            kind: "info",
            message: "This profile is in the old format, with no dates. The next profile batch rebuilds it.",
          })
        : null,
      profile.version === 0
        ? D.PanelState({ kind: "info", message: "This profile has no structured data, only the text below." })
        : null,
      raw
        ? e(
            "pre",
            { className: "profile-raw mono" },
            JSON.stringify(Object.assign({}, profile, { summary: undefined }), null, 2)
          )
        : null,
      profile.overview ? e("p", { className: "profile-overview" }, profile.overview) : null,
      profile.communication_style
        ? e(
            "p",
            { className: "profile-style" },
            e("span", { className: "profile-style-label" }, isSelf ? "How it talks" : "How they talk"),
            profile.communication_style
          )
        : null,
      profile.version === 0 && profile.summary ? e("div", { className: "profile-plain" }, profile.summary) : null,
      filled.length
        ? e(
            "div",
            { className: "profile-nav" },
            e(
              "div",
              { className: "profile-chips" },
              e(
                "button",
                {
                  type: "button",
                  className: "profile-chip" + (section === "all" ? " profile-chip--on" : ""),
                  "aria-pressed": section === "all" ? "true" : "false",
                  onClick: function () {
                    setSection("all");
                  },
                },
                "All"
              ),
              filled.map(function (sec) {
                var on = section === sec.key;
                return e(
                  "button",
                  {
                    key: sec.key,
                    type: "button",
                    className: "profile-chip" + (on ? " profile-chip--on" : "") + (sec.shown.length ? "" : " profile-chip--dim"),
                    "aria-pressed": on ? "true" : "false",
                    title: sec.guide || null,
                    onClick: function () {
                      setSection(on ? "all" : sec.key);
                    },
                  },
                  sec.label,
                  e("span", { className: "profile-chip-count" }, filtering ? sec.shown.length + "/" + sec.items.length : sec.items.length)
                );
              })
            ),
            e("input", {
              type: "search",
              className: "profile-filter",
              placeholder: "Filter this profile…",
              "aria-label": "Filter this profile",
              value: filter,
              onChange: function (ev) {
                setFilter(ev.target.value);
              },
            })
          )
        : null,
      filtering && !matchCount ? D.PanelState({ kind: "empty", message: "Nothing in this profile matches “" + filter.trim() + "”." }) : null,
      visible.map(function (sec) {
        return e(
          "section",
          { key: sec.key, className: "profile-section" },
          e(
            "h5",
            { className: "profile-section-title", title: sec.guide || null },
            sec.label,
            e("span", { className: "profile-section-count" }, sec.cap ? sec.items.length + " of " + sec.cap : sec.items.length)
          ),
          e(
            "ul",
            { className: "profile-items" },
            sec.shown.map(function (it) {
              var when = itemWhen(it);
              return e(
                "li",
                { key: it.id, className: "profile-item" },
                e(
                  "span",
                  { className: "profile-item-text" },
                  it.name ? e("strong", { className: "profile-item-name" }, it.name + ": ") : null,
                  highlight(it.text, filter)
                ),
                when ? e("span", { className: "profile-item-when", title: itemWhenTitle(it) }, when) : null
              );
            })
          )
        );
      }),
      empty.length && !filtering && section === "all" && profile.version === 2
        ? e(
            "p",
            { className: "profile-empty muted" },
            "Nothing yet under: " +
              empty
                .map(function (sec) {
                  return sec.label;
                })
                .join(", ")
          )
        : null
    );
  }

  function ProfilesView({ guild, epoch, search, onSearch, selectedId, onSelect, onShowMessages, onOpenJobs }) {
    const [rows, setRows] = useState([]);
    const [total, setTotal] = useState(0);
    const [status, setStatus] = useState("loading");
    const term = D.useDebounced(search, 300);
    const ticket = useRef(0);

    useEffect(
      function () {
        if (!guild) return;
        var mine = ++ticket.current;
        setStatus("loading");
        var url = guildUrl("/api/profiles/list/", guild) + (term.trim() ? "?search=" + encodeURIComponent(term.trim()) : "");
        D.readJson(url)
          .then(function (data) {
            if (mine !== ticket.current) return;
            // Soupy's own memory sits first, set apart from the members.
            setRows((data.self ? [data.self] : []).concat(data.rows || []));
            setTotal(Number(data.total || 0));
            setStatus("ready");
          })
          .catch(function () {
            if (mine !== ticket.current) return;
            setRows([]);
            setStatus("error");
          });
      },
      [guild, term, epoch]
    );

    // Keep something open: the most-read member, or whoever still matches the search.
    useEffect(
      function () {
        if (status !== "ready") return;
        var listed = rows.some(function (r) {
          return r.user_id === selectedId;
        });
        if (listed) return;
        var first =
          rows.filter(function (r) {
            return r.kind !== "self";
          })[0] || rows[0];
        onSelect(first ? first.user_id : null);
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [rows, status]
    );

    var searching = !!term.trim();

    if (status === "ready" && !rows.length && !searching) {
      return e(
        "div",
        { className: "db-pane" },
        D.PanelState({
          kind: "empty",
          message: "No profiles for this server yet.",
          children: e("button", { type: "button", className: "secondary", onClick: onOpenJobs }, "Open the profile job"),
        })
      );
    }

    return e(
      "div",
      { className: "db-pane profiles" },
      e(
        "aside",
        { className: "profiles-list" },
        e("input", {
          type: "search",
          className: "profiles-search",
          placeholder: "Search names and profile text…",
          "aria-label": "Search profiles",
          value: search,
          onChange: function (ev) {
            onSearch(ev.target.value);
          },
        }),
        e(
          "div",
          { className: "profiles-count muted" },
          status === "error"
            ? "Could not load profiles."
            : searching
              ? D.plural(rows.length, "match", "matches") + " in " + D.plural(total, "profile")
              : D.plural(total, "member profile")
        ),
        e(
          "div",
          { className: "profiles-rows", role: "listbox", "aria-label": "Profiles" },
          rows.map(function (r) {
            var on = r.user_id === selectedId;
            var pick = function () {
              onSelect(r.user_id);
            };
            return e(
              "div",
              {
                key: r.user_id,
                className: "profiles-row" + (on ? " profiles-row--on" : "") + (r.kind === "self" ? " profiles-row--self" : ""),
                role: "option",
                "aria-selected": on ? "true" : "false",
                tabIndex: 0,
                onClick: pick,
                onKeyDown: activate(pick),
              },
              e("div", { className: "profiles-row-name" }, r.name, r.version !== 2 ? e("span", { className: "pill" }, "old format") : null),
              e(
                "div",
                { className: "profiles-row-meta" },
                r.kind === "self"
                  ? "its own memory · " + D.plural(r.item_count, "item")
                  : D.plural(r.message_count, "msg") + " · " + D.plural(r.item_count, "item")
              )
            );
          })
        )
      ),
      e(
        "div",
        { className: "profiles-detail" },
        selectedId
          ? e(ProfileDetail, { guild: guild, id: selectedId, term: term, epoch: epoch, onShowMessages: onShowMessages })
          : status === "ready"
            ? D.PanelState({ kind: "empty", message: "No profile matches “" + term.trim() + "”." })
            : D.PanelState({ kind: "loading", message: "Loading profiles…" })
      )
    );
  }

  // ---------------------------------------------------------------------------
  // Maintenance
  // ---------------------------------------------------------------------------

  /** One line of the profile job's log, with its timestamp split off. */
  function logLine(line, i) {
    var s = String(line || "");
    var m = /^\[([^\]]+)\]\s*(.*)$/.exec(s);
    if (!m) return e("div", { key: i, className: "job-log-row job-log-row--raw" }, s);
    return e(
      "div",
      { key: i, className: "job-log-row" },
      e("span", { className: "job-log-when", title: m[1] }, D.fmtDateTime(m[1])),
      e("span", { className: "job-log-text" }, m[2])
    );
  }

  function batchSummary(batch, botRunning) {
    var parts = [];
    if (batch.kind && batch.kind !== "manual") parts.push(batch.kind + " run");
    var total = Number(batch.total || 0);
    if (total) parts.push(D.fmtNum(Math.min(Number(batch.next_index || 0), total)) + " of " + D.plural(total, "member"));
    var stats = batch.batch_stats;
    if (stats && stats.saved != null) {
      parts.push(D.fmtNum(stats.saved) + " saved · " + D.fmtNum(stats.skipped || 0) + " skipped · " + D.fmtNum(stats.failed || 0) + " failed");
    }
    if (batch.status === "running") {
      if (!botRunning) parts.push("on hold: the bot is stopped");
      else if (batch.waiting_for_bot) parts.push("queued until the bot is free");
    }
    return parts.join(" · ");
  }

  function MaintenanceView(props) {
    const { guild, db, batch, batchLog, botRunning, busy, notify, confirm, ragStatus } = props;
    const [minutes, setMinutes] = useState("0");
    const [scans, setScans] = useState({ rows: [], total: 0 });
    const logRef = useRef(null);
    const interval = db ? db.archive_scan_interval_minutes : null;
    const lastScan = db ? db.last_scan : null;

    useEffect(
      function () {
        setMinutes(String(interval != null ? interval : 0));
      },
      [guild, interval]
    );

    useEffect(
      function () {
        if (!guild) return;
        var cancelled = false;
        D.readJson(guildUrl("/api/database/explore/", guild) + "?table=scan_metadata&limit=6")
          .then(function (data) {
            if (!cancelled) setScans({ rows: data.rows || [], total: Number(data.total || 0) });
          })
          .catch(function () {});
        return function () {
          cancelled = true;
        };
      },
      [guild, lastScan]
    );

    useEffect(
      function () {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
      },
      [batchLog]
    );

    async function jobAction(action, message) {
      try {
        var data = await D.postJson(guildUrl("/api/profiles/batch/" + action + "/", guild));
        if (message) notify(typeof message === "function" ? message(data) : message, false);
        props.onJobChanged();
      } catch (err) {
        notify("Profile job: " + err.message, true);
      }
    }

    async function startBatch() {
      var ok = await confirm({
        title: "Start a profile batch?",
        message:
          "The bot builds each member's profile in several passes with the chat model, one call at a time, between chat replies. It can be paused, resumed and cancelled.",
        confirmLabel: "Start batch",
      });
      if (ok) {
        jobAction("start", function (data) {
          return "Profile batch queued for " + D.plural(data.total || 0, "member") + ".";
        });
      }
    }

    async function cancelBatch() {
      var ok = await confirm({
        title: "Cancel the profile job?",
        message: "The bot stops after the pass it is on. Everything saved so far is kept.",
        confirmLabel: "Cancel job",
        cancelLabel: "Keep running",
        danger: true,
      });
      if (ok) jobAction("cancel", "Cancel requested.");
    }

    async function deleteProfiles() {
      var ok = await confirm({
        title: "Delete every profile for this server?",
        message: "All stored member profiles and the job's state are removed. A new batch rebuilds them from the archive, which takes hours on a busy server.",
        confirmLabel: "Delete all profiles",
        danger: true,
      });
      if (!ok) return;
      try {
        await D.postJson(guildUrl("/api/profiles/reset/", guild));
        notify("All profiles deleted for this server.", false);
        props.onJobChanged();
        props.onProfilesChanged();
      } catch (err) {
        notify("Delete failed: " + err.message, true);
      }
    }

    async function saveSchedule() {
      var m = parseInt(String(minutes).trim(), 10);
      if (isNaN(m) || m < 0 || m > 10080) {
        notify("Minutes between scans must be 0 to 10080 (7 days). 0 turns automatic scans off.", true);
        return;
      }
      try {
        var data = await D.postJson(guildUrl("/api/database/archive-schedule/", guild), { minutes: m });
        notify(
          data.archive_scan_interval_minutes === 0
            ? "Automatic scans are off for this server."
            : "This server is scanned every " + D.plural(data.archive_scan_interval_minutes, "minute") + ".",
          false
        );
        props.onReloadDatabases();
      } catch (err) {
        notify("Could not save the schedule: " + err.message, true);
      }
    }

    var state = batch ? String(batch.status || "idle") : "…";
    var live = !!batch && (batch.task_running || state === "running");
    var paused = state === "paused";
    var total = batch ? Number(batch.total || 0) : 0;
    var done = batch ? Math.min(Number(batch.next_index || 0), total) : 0;

    return e(
      "div",
      { className: "db-pane jobs" },
      e(
        "section",
        { className: "job-card job-card--wide" },
        e(
          "div",
          { className: "job-card-head" },
          e("h5", null, "Member profiles"),
          e("span", { className: "job-chip" + (live ? " job-chip--live" : paused ? " job-chip--paused" : "") }, state)
        ),
        e(
          "p",
          { className: "job-card-note muted" },
          "Built from the archive in dated passes and refreshed nightly from new messages. The bot runs the job; this page queues and steers it."
        ),
        batch ? e("div", { className: "job-summary" }, batchSummary(batch, botRunning) || "No job has run yet.") : null,
        total && (live || paused)
          ? e("div", { className: "job-progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": total, "aria-valuenow": done },
              e("div", { className: "job-progress-fill", style: { width: (done / total) * 100 + "%" } }))
          : null,
        e(
          "div",
          { className: "job-actions" },
          e("button", { type: "button", disabled: busy || live || paused, onClick: startBatch }, "Start batch"),
          e(
            "button",
            {
              type: "button",
              className: "secondary",
              disabled: state !== "running",
              onClick: function () {
                jobAction("pause", "Pausing after the current pass.");
              },
            },
            "Pause"
          ),
          e(
            "button",
            {
              type: "button",
              className: "secondary",
              disabled: !paused,
              onClick: function () {
                jobAction("resume", "Resumed.");
              },
            },
            "Resume"
          ),
          e("button", { type: "button", className: "secondary", disabled: !(live || paused), onClick: cancelBatch }, "Cancel"),
          e("span", { className: "job-actions-gap" }),
          e("button", { type: "button", className: "dash-btn-danger-outline", disabled: busy || live, onClick: deleteProfiles }, "Delete all profiles")
        ),
        e(
          "div",
          { className: "job-log mono", ref: logRef, role: "log", "aria-live": "polite", "aria-relevant": "additions" },
          batchLog.length
            ? batchLog.map(logLine)
            : e("div", { className: "job-log-empty" }, "The job's log shows here: each pass it reads, what it sent the model, what it saved.")
        )
      ),
      e(
        "section",
        { className: "job-card" },
        e("div", { className: "job-card-head" }, e("h5", null, "Message archive")),
        e(
          "div",
          { className: "job-summary" },
          lastScan ? "Last scan " + D.timeAgo(lastScan) : "Never scanned",
          scans.total ? " · " + D.plural(scans.total, "scan") + " so far" : ""
        ),
        e(
          "div",
          { className: "job-actions" },
          e("button", { type: "button", disabled: busy || !botRunning, title: botRunning ? null : "Start the bot first", onClick: props.onScan }, "Scan now"),
          e(
            "label",
            { className: "job-schedule" },
            "Scan every",
            e("input", {
              type: "number",
              min: 0,
              max: 10080,
              step: 1,
              value: minutes,
              onChange: function (ev) {
                setMinutes(ev.target.value);
              },
            }),
            "minutes"
          ),
          e("button", { type: "button", className: "secondary", disabled: String(interval) === String(minutes).trim(), onClick: saveSchedule }, "Save")
        ),
        e("p", { className: "job-card-note muted" }, "0 turns automatic scans off. The search index is rebuilt after each scan that finds new messages."),
        scans.rows.length
          ? e(
              "div",
              { className: "job-scans" },
              scans.rows.map(function (s, i) {
                return e(
                  "div",
                  { key: i, className: "job-scan" },
                  e("span", { title: D.fmtDateTimeFull(s.last_scan_time) }, D.timeAgo(s.last_scan_time)),
                  e("span", { className: "muted" }, s.scan_type || ""),
                  e("span", { className: "job-scan-count" }, "+" + D.fmtNum(s.messages_scanned))
                );
              })
            )
          : null
      ),
      e(
        "section",
        { className: "job-card" },
        e("div", { className: "job-card-head" }, e("h5", null, "Search index (RAG)")),
        e(
          "div",
          { className: "job-summary" },
          ragStatus && ragStatus.ok
            ? D.plural(ragStatus.rag_chunks || 0, "chunk") + " indexed from " + D.plural(ragStatus.total_messages || 0, "message")
            : "Index status unavailable."
        ),
        e(
          "div",
          { className: "job-actions" },
          e("button", { type: "button", className: "secondary", disabled: busy, onClick: props.onRebuildRag }, "Rebuild index")
        ),
        e("p", { className: "job-card-note muted" }, "A rebuild re-embeds every message through the embedding model in LM Studio, so it has to be loaded.")
      )
    );
  }

  // ---------------------------------------------------------------------------
  // Tab
  // ---------------------------------------------------------------------------

  function DatabaseTab(props) {
    const { active, guild, databases, view, botRunning } = props;
    // Held here, above the three views, so a filter or an open profile is
    // still there after a look at another view or another tab.
    const [filters, setFilters] = useScoped(guild, EMPTY_FILTERS);
    const [profileSearch, setProfileSearch] = useScoped(guild, "");
    const [profileId, setProfileId] = useScoped(guild, null);
    const [batch, setBatch] = useState(null);
    const [batchLog, setBatchLog] = useState([]);
    const [pollEpoch, setPollEpoch] = useState(0);
    const [profilesEpoch, setProfilesEpoch] = useState(0);

    // Follow the profile job while this tab is open. When it saves a profile
    // (the row count or its place in the queue moves), the profile list reloads.
    useEffect(
      function () {
        setBatch(null);
        setBatchLog([]);
        if (!active || !guild) return;
        var cancelled = false;
        var timer = null;
        var lastSync = null;

        async function tick() {
          var delay = BATCH_POLL_IDLE_MS;
          try {
            var data = await D.readJson(guildUrl("/api/profiles/batch/status/", guild));
            if (cancelled) return;
            setBatch(data);
            setBatchLog(Array.isArray(data.log_lines) ? data.log_lines : []);
            var sync = Number(data.next_index || 0) + ":" + Number(data.profile_row_count || 0);
            if (lastSync !== null && sync !== lastSync) {
              setProfilesEpoch(function (n) {
                return n + 1;
              });
            }
            lastSync = sync;
            var active = data.task_running || data.status === "running" || data.status === "paused";
            delay = active ? BATCH_POLL_ACTIVE_MS : BATCH_POLL_IDLE_MS;
          } catch (_err) {
            if (cancelled) return;
            delay = BATCH_POLL_ERR_MS;
          }
          timer = setTimeout(tick, delay);
        }
        tick();
        return function () {
          cancelled = true;
          if (timer != null) clearTimeout(timer);
        };
      },
      [active, guild, pollEpoch]
    );

    if (!active) return null;

    if (!databases.length) {
      return e(
        "article",
        { className: "db-tab" },
        D.PanelState({
          kind: "empty",
          message: "No server has been archived yet. Run /soupyscan in Discord as an owner, then come back here.",
        })
      );
    }

    var db = databases.filter(function (d) {
      return String(d.guild_id) === String(guild);
    })[0];
    var jobLive = !!batch && (batch.task_running || batch.status === "running");

    var stats = [];
    if (db) {
      stats.push(D.plural(db.total_messages, "message"));
      stats.push(D.plural(db.total_channels, "channel"));
      if (batch && batch.profile_row_count != null) stats.push(D.plural(batch.profile_row_count, "profile"));
      stats.push(db.last_scan ? "scanned " + D.timeAgo(db.last_scan) : "never scanned");
      stats.push(D.fmtBytes(db.file_size));
    }

    return e(
      "article",
      { className: "db-tab" },
      e(
        "div",
        { className: "db-head" },
        e(
          "label",
          { className: "db-head-server" },
          "Server",
          e(
            "select",
            {
              value: guild,
              onChange: function (ev) {
                props.onGuild(ev.target.value);
              },
            },
            databases.map(function (d) {
              return e("option", { key: d.guild_id, value: d.guild_id }, d.guild_name || "Server " + d.guild_id);
            })
          )
        ),
        e("div", { className: "db-head-stats muted" }, stats.join(" · ")),
        e(D.Segmented, {
          label: "Database view",
          value: view,
          onChange: props.onView,
          options: [
            { value: "messages", label: "Messages" },
            { value: "profiles", label: "Profiles" },
            { value: "maintenance", label: "Maintenance", dot: jobLive ? "A profile job is running" : null },
          ],
        })
      ),
      view === "messages" ? e(MessagesView, { key: guild, guild: guild, filters: filters, onFilters: setFilters }) : null,
      view === "profiles"
        ? e(ProfilesView, {
            key: guild,
            guild: guild,
            epoch: profilesEpoch,
            search: profileSearch,
            onSearch: setProfileSearch,
            selectedId: profileId,
            onSelect: setProfileId,
            onShowMessages: function (userId, name) {
              setFilters(Object.assign({}, EMPTY_FILTERS, { member: userId, memberName: name }));
              props.onView("messages");
            },
            onOpenJobs: function () {
              props.onView("maintenance");
            },
          })
        : null,
      view === "maintenance"
        ? e(MaintenanceView, {
            key: guild,
            guild: guild,
            db: db,
            batch: batch,
            batchLog: batchLog,
            botRunning: botRunning,
            busy: props.busy,
            notify: props.notify,
            confirm: props.confirm,
            ragStatus: props.ragStatus,
            onRebuildRag: props.onRebuildRag,
            onScan: props.onScan,
            onReloadDatabases: props.onReloadDatabases,
            onJobChanged: function () {
              setPollEpoch(function (n) {
                return n + 1;
              });
            },
            onProfilesChanged: function () {
              setProfilesEpoch(function (n) {
                return n + 1;
              });
            },
          })
        : null
    );
  }

  D.DatabaseTab = DatabaseTab;
})();
