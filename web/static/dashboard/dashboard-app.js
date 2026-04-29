(function () {
  const e = React.createElement;
  const { useCallback, useEffect, useMemo, useState, useRef } = React;

  function fmtNum(v) {
    return Number(v || 0).toLocaleString();
  }

  /** One line in the profile-batch activity console (timestamp split for readability). */
  function profileBatchLogLineRow(line, idx) {
    var s = String(line || "");
    var m = /^\[([^\]]+)\]\s*(.*)$/.exec(s);
    if (m) {
      return e(
        "div",
        { key: "pbl-" + idx, className: "profile-batch-log-row" },
        e("span", { className: "profile-batch-log-ts" }, m[1]),
        e("span", { className: "profile-batch-log-msg" }, m[2])
      );
    }
    return e("div", { key: "pbl-" + idx, className: "profile-batch-log-row profile-batch-log-row-raw" }, s);
  }

  function previewText(text, maxLen) {
    var s = String(text || "").replace(/\s+/g, " ").trim();
    if (s.length <= maxLen) return s;
    return s.slice(0, Math.max(0, maxLen - 1)) + "…";
  }

  function pickerSelectValue(userList, currentId) {
    var s = String(currentId || "").trim();
    if (!s) return "";
    var found = (userList || []).some(function (u) {
      return String(u.user_id) === s;
    });
    return found ? s : "";
  }

  function normalizeLineEndings(text) {
    return String(text || "").replace(/\r\n?/g, "\n");
  }

  /**
   * Turn dense one-line behaviours (as stored in .env) into readable, editable paragraphs
   * in the browser: section bars, numbered lists, glued "2. …" items, ALL‑CAPS headers, etc.
   */
  function formatBehaviourForEditor(text) {
    var t = normalizeLineEndings(text).trim();
    if (!t) return t;

    var pass;
    for (pass = 0; pass < 5; pass++) {
      var before = t;

      // Long equals bars with a title between them
      t = t.replace(/={12,}([^=\n][^=\n]*?)={12,}/g, "\n\n$1\n\n");

      // Double-space after sentence end → paragraph (common in your prompts)
      t = t.replace(/([.!?])\s{2,}(?=[A-Z*0-9])/g, "$1\n\n");

      // "Section title:" immediately followed by "1." (no newline)
      t = t.replace(/:([1-9]\d{0,2}\.\s)/g, ":\n\n$1");

      // "1.You" / "10.Behave" → space after the dot (lists crushed in .env)
      t = t.replace(/([1-9]\d{0,2}\.)([A-Za-z*])/g, "$1 $2");

      // Word / punctuation glued to next list number: "…thing2. You" / "…topic10. Behave"
      t = t.replace(/([a-z.!?)])([1-9]\d{0,2}\.\s)/gi, "$1\n\n$2");
      t = t.replace(/([a-z.!?)])([1-9]\d{0,2}\.)(?=[A-Za-z*])/gi, "$1\n\n$2 ");

      // ALL‑CAPS word + period + Capitalized word: "IMPORTANT.Core" / "RESPONSE.Your"
      t = t.replace(/([A-Z]{2,})\.([A-Z][a-z]+)/g, "$1.\n\n$2");

      // ",INCLUDING" / instruction boilerplate
      t = t.replace(/,(\s*)(INCLUDING\b)/gi, ",\n\n$2");

      // Emphasis blocks in older prompts
      t = t.replace(/\*\*([^*]+)\*\*/g, "\n\n**$1**\n\n");

      t = t.replace(/\n{3,}/g, "\n\n");
      if (t === before) break;
    }

    // Final sentence + long ALL‑CAPS clause (e.g. trailing NEVER EVER…)
    t = t.replace(/([.!?])\s+([A-Z][A-Z\s,\d]{18,})\s*$/m, "$1\n\n$2");

    t = t.replace(/\n{3,}/g, "\n\n");
    return t.trim();
  }

  function parseModelList(raw) {
    return String(raw || "")
      .split(/[\n,]/)
      .map(function (m) {
        return m.trim();
      })
      .filter(Boolean);
  }

  function uniqueModels(list) {
    const seen = new Set();
    const out = [];
    list.forEach(function (item) {
      const key = String(item || "").trim();
      if (!key) return;
      const low = key.toLowerCase();
      if (!seen.has(low)) {
        seen.add(low);
        out.push(key);
      }
    });
    return out;
  }

  function titleFromPersonalityKey(key) {
    if (key === "BEHAVIOUR") return "Live chat prompt (env)";
    if (key === "INTERJECT") return "Interject personality";
    if (key === "9BALL") return "9-ball personality";
    return key
      .replace(/^BEHAVIOUR_?/, "")
      .replace(/^PERSONALITY_?/, "")
      .split("_")
      .filter(Boolean)
      .map(function (part) {
        return part.charAt(0) + part.slice(1).toLowerCase();
      })
      .join(" ");
  }

  async function readJson(url, init) {
    const res = await fetch(url, init);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.message || data.error || "Request failed");
    }
    return data;
  }

  function SparkLine({ points, labels, color, compact, title }) {
    if (!points || !points.length) return e("div", { className: "muted" }, "No trend data.");
    const vbW = 560;
    const vbH = 184;
    const padX = 14;
    const padTop = 10;
    const labelBand = 26;
    const chartBottom = vbH - labelBand;
    const chartH = chartBottom - padTop;
    const min = Math.min.apply(null, points);
    const max = Math.max.apply(null, points);
    const span = max - min || 1;
    const chartW = vbW - 2 * padX;
    const dx = chartW / Math.max(1, points.length - 1);
    const path = points
      .map((p, i) => {
        const x = padX + i * dx;
        const y = chartBottom - ((p - min) / span) * (chartH - 4) - 2;
        return (i ? "L" : "M") + x + "," + y;
      })
      .join(" ");
    return e(
      "div",
      { className: "spark" + (compact ? " spark--compact" : "") },
      title ? e("div", { className: "stats-chart-title" }, title) : null,
      e(
        "svg",
        {
          className: "spark-svg",
          viewBox: "0 0 " + vbW + " " + vbH,
          preserveAspectRatio: "xMidYMid meet",
          "aria-hidden": true,
        },
        e("path", { d: path, fill: "none", stroke: color || "#7aa2ff", strokeWidth: 2.2, vectorEffect: "non-scaling-stroke" }),
        points.map((p, i) => {
          const x = padX + i * dx;
          const y = chartBottom - ((p - min) / span) * (chartH - 4) - 2;
          return e("circle", { key: i, cx: x, cy: y, r: 2.2, fill: color || "#7aa2ff" });
        }),
        (labels || []).map((label, i) => {
          var nl = (labels || []).length;
          var anchor =
            nl <= 1 ? "middle" : i === 0 ? "start" : i === nl - 1 ? "end" : "middle";
          return e(
            "text",
            {
              key: "l" + i,
              x: padX + i * dx,
              y: vbH - 6,
              fontSize: "10",
              fill: "var(--text-muted)",
              textAnchor: anchor,
            },
            String(label).length > 5 ? String(label).slice(5) : String(label)
          );
        })
      )
    );
  }

  function NormalizedMultiLine({ labels, series, title }) {
    var active = (series || []).filter(function (s) {
      return s.enabled && s.values && s.values.length;
    });
    if (!labels || !labels.length || !active.length) {
      return e("div", { className: "muted" }, title ? "No data for " + title : "No multi-series data.");
    }
    var vbW = 560;
    var vbH = 192;
    var padX = 14;
    var padTop = 8;
    var labelBand = 26;
    var chartBottom = vbH - labelBand;
    var chartH = chartBottom - padTop;
    var n = labels.length;
    var chartW = vbW - 2 * padX;
    var dx = chartW / Math.max(1, n - 1);
    function normPath(vals) {
      var min = Math.min.apply(null, vals);
      var max = Math.max.apply(null, vals);
      var span = max - min || 1;
      return vals
        .map(function (p, i) {
          var x = padX + i * dx;
          var y = chartBottom - ((p - min) / span) * (chartH - 4) - 2;
          return (i ? "L" : "M") + x + "," + y;
        })
        .join(" ");
    }
    return e(
      "div",
      { className: "spark" },
      title ? e("div", { className: "stats-chart-title" }, title) : null,
      e(
        "svg",
        {
          className: "spark-svg",
          viewBox: "0 0 " + vbW + " " + vbH,
          preserveAspectRatio: "xMidYMid meet",
          "aria-hidden": true,
        },
        active.map(function (s, si) {
          return e("path", {
            key: s.key || si,
            d: normPath(s.values),
            fill: "none",
            stroke: s.color,
            strokeWidth: 2,
            opacity: 0.92,
            vectorEffect: "non-scaling-stroke",
          });
        }),
        labels.map(function (label, i) {
          var anchor =
            n <= 1 ? "middle" : i === 0 ? "start" : i === n - 1 ? "end" : "middle";
          return e(
            "text",
            {
              key: "nl" + i,
              x: padX + i * dx,
              y: vbH - 6,
              fontSize: "9",
              fill: "var(--text-muted)",
              textAnchor: anchor,
            },
            String(label).length > 5 ? String(label).slice(5) : String(label)
          );
        })
      )
    );
  }

  function HourlyStackedBars({ labels, totals, generated, analyzed, title, archiveVolumeOnly }) {
    var n = 24;
    var width = 560;
    var height = 200;
    var msgs = totals || [];
    var gen = generated || [];
    var vis = analyzed || [];
    var volumeOnly = !!archiveVolumeOnly;
    var stackedMax = 1;
    var i;
    for (i = 0; i < n; i++) {
      var g = gen[i] || 0;
      var v = vis[i] || 0;
      var o = Math.max(0, (msgs[i] || 0) - g - v);
      stackedMax = Math.max(stackedMax, g + v + o);
    }
    var slot = width / n;
    var barW = Math.max(4, slot * 0.62);
    var baseY = height - 22;
    var chartH = height - 34;
    var els = [];
    for (i = 0; i < n; i++) {
      var g0 = volumeOnly ? 0 : gen[i] || 0;
      var v0 = volumeOnly ? 0 : vis[i] || 0;
      var o0 = volumeOnly ? (msgs[i] || 0) : Math.max(0, (msgs[i] || 0) - g0 - v0);
      var x = i * slot + (slot - barW) / 2;
      var y = baseY;
      var h2 = (g0 / stackedMax) * chartH;
      if (h2 > 0) {
        els.push(
          e("rect", {
            key: "g" + i,
            x: x,
            y: y - h2,
            width: barW,
            height: h2,
            fill: "color-mix(in srgb, #9ece6a 85%, white 15%)",
            rx: 1.5,
          })
        );
        y -= h2;
      }
      h2 = (v0 / stackedMax) * chartH;
      if (h2 > 0) {
        els.push(
          e("rect", {
            key: "v" + i,
            x: x,
            y: y - h2,
            width: barW,
            height: h2,
            fill: "color-mix(in srgb, #bb9af7 80%, white 20%)",
            rx: 1.5,
          })
        );
        y -= h2;
      }
      h2 = (o0 / stackedMax) * chartH;
      if (h2 > 0) {
        els.push(
          e("rect", {
            key: "o" + i,
            x: x,
            y: y - h2,
            width: barW,
            height: h2,
            fill: "color-mix(in srgb, #5fa8ff 75%, white 25%)",
            rx: 1.5,
          })
        );
      }
    }
    var labelEls = [];
    for (var j = 0; j < 6; j++) {
      var idx = j * 4;
      if (idx >= n) break;
      var lab = (labels[idx] || "").toString().slice(0, 9);
      var lx = idx * slot + slot / 2;
      labelEls.push(
        e(
          "text",
          {
            key: "hb" + idx,
            x: lx,
            y: height - 5,
            fontSize: "8",
            textAnchor: "middle",
            fill: "var(--text-muted)",
          },
          lab
        )
      );
    }
    return e(
      "div",
      { className: "spark" },
      title ? e("div", { className: "stats-chart-title" }, title) : null,
      e(
        "svg",
        {
          className: "spark-svg",
          viewBox: "0 0 " + width + " " + height,
          preserveAspectRatio: "xMidYMid meet",
          "aria-hidden": true,
        },
        els.concat(labelEls)
      ),
      volumeOnly
        ? e(
            "div",
            { className: "stats-legend stats-legend--compact" },
            e(
              "span",
              { className: "stats-legend-item" },
              e("i", { style: { background: "#5fa8ff" } }),
              " messages (scanned archive)"
            )
          )
        : e(
            "div",
            { className: "stats-legend stats-legend--compact" },
            e("span", { className: "stats-legend-item" }, e("i", { style: { background: "#9ece6a" } }), " generated"),
            e("span", { className: "stats-legend-item" }, e("i", { style: { background: "#bb9af7" } }), " vision"),
            e("span", { className: "stats-legend-item" }, e("i", { style: { background: "#5fa8ff" } }), " other msgs")
          )
    );
  }

  function LeaderBoard({ title, rows, valueKey, labelKey, fallback }) {
    var r = rows || [];
    var maxVal = Math.max.apply(
      null,
      [1].concat(
        r.map(function (x) {
          return Number(x[valueKey] || 0);
        })
      )
    );
    return e(
      "article",
      { className: "stats-leader-card" },
      e("h5", { style: { marginTop: 0 } }, title),
      !r.length
        ? e("p", { className: "muted" }, "No data")
        : e(
            "div",
            { className: "leader-list" },
            r.map(function (row, i) {
              var v = Number(row[valueKey] || 0);
              var pct = (v / maxVal) * 100;
              var label = row[labelKey] || row.channel_name || row.channel_id || fallback || "Unknown";
              return e(
                "div",
                { className: "leader-row leader-row-bar", key: i },
                e("div", { className: "leader-bar-track" }, e("div", { className: "leader-bar-fill", style: { width: pct + "%" } })),
                e(
                  "div",
                  { className: "leader-row-inner" },
                  e("div", null, label),
                  e("div", { className: "leader-pill mono" }, fmtNum(v))
                )
              );
            })
          )
    );
  }

  function StatsStudio({ summary, active, onReload }) {
    const [live, setLive] = useState(false);
    const [pollSec, setPollSec] = useState(30);
    const [seriesMask, setSeriesMask] = useState({
      messages: true,
      gen: true,
      vis: true,
      other: true,
    });

    useEffect(
      function () {
        if (!active || !live) return;
        onReload();
        var id = window.setInterval(onReload, pollSec * 1000);
        return function () {
          window.clearInterval(id);
        };
      },
      [active, live, pollSec, onReload]
    );

    var s7 =
      summary && summary.series_7d
        ? summary.series_7d
        : { days: [], messages: [], images_generated: [], images_analyzed: [], other_messages: [] };
    var dayLabels = s7.days || [];
    var hourly =
      summary && summary.hourly_24h
        ? summary.hourly_24h
        : { labels: [], messages: [], images_generated: [], images_analyzed: [] };
    var ev = (summary && summary.event_totals) || {};

    function toggleMask(k) {
      setSeriesMask(function (prev) {
        var next = Object.assign({}, prev);
        next[k] = !prev[k];
        return next;
      });
    }

    var multiSeries = [
      {
        key: "m",
        label: "All messages",
        color: "#5fa8ff",
        values: s7.messages || [],
        enabled: seriesMask.messages,
      },
      {
        key: "g",
        label: "Image generation",
        color: "#9ece6a",
        values: s7.images_generated || [],
        enabled: seriesMask.gen,
      },
      {
        key: "v",
        label: "Vision",
        color: "#bb9af7",
        values: s7.images_analyzed || [],
        enabled: seriesMask.vis,
      },
      {
        key: "o",
        label: "Other chat",
        color: "#e0af68",
        values: s7.other_messages || [],
        enabled: seriesMask.other,
      },
    ];

    var eventPairs = Object.keys(ev)
      .map(function (k) {
        return { k: k, n: Number(ev[k] || 0) };
      })
      .sort(function (a, b) {
        return b.n - a.n;
      })
      .slice(0, 14);

    var fromSqliteArchive =
      summary &&
      summary.stats_source &&
      summary.stats_source.message_counts === "sqlite_archive";

    return e(
      "div",
      { className: "stats-studio" },
      e(
        "div",
        { className: "stats-studio-toolbar" },
        e(
          "button",
          {
            type: "button",
            className: live ? "" : "secondary",
            onClick: function () {
              setLive(!live);
            },
          },
          live ? "Live refresh on" : "Live refresh off"
        ),
        e(
          "label",
          { className: "stats-live-gap" },
          "Every",
          e(
            "select",
            {
              value: String(pollSec),
              onChange: function (ev) {
                setPollSec(Number(ev.target.value || 30));
              },
            },
            [5, 10, 15, 30, 60, 120].map(function (n) {
              return e("option", { key: n, value: String(n) }, n + "s");
            })
          )
        ),
        e(
          "button",
          { type: "button", className: "secondary", onClick: onReload },
          "Refresh now"
        ),
        live
          ? e("span", { className: "muted stats-live-hint" }, "Polling /api/stats/summary while this tab is open.")
          : null
      ),
      e(
        "p",
        { className: "muted", style: { marginTop: 0 } },
        fromSqliteArchive
          ? "Message totals, channel leaderboards (with real channel names), 7‑day volume, and the 24‑hour chart are driven by your scanned Discord archive (per‑guild SQLite). Thin green/purple lines are still bot JSONL events; “event mix” is JSONL only."
          : "Counts come from media/messages.jsonl unless you have scanned archive databases under soupy_database/databases. Line chart uses independent scales per series so you can compare timing, not absolute volume."
      ),
      e(
        "div",
        { className: "dash-kpis", style: { marginBottom: "0.85rem" } },
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "messages (total)"),
          e("div", { className: "dash-kpi-value" }, fmtNum(summary && summary.totals ? summary.totals.messages : 0))
        ),
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "msgs last 24h"),
          e("div", { className: "dash-kpi-value" }, fmtNum(summary ? summary.last24_messages : 0))
        ),
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "msgs last 48h"),
          e("div", { className: "dash-kpi-value" }, fmtNum(summary ? summary.last48_messages : 0))
        ),
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "msgs 7d / 30d"),
          e(
            "div",
            { className: "dash-kpi-value" },
            fmtNum(summary ? summary.last7d_messages : 0) + " / " + fmtNum(summary ? summary.last30d_messages : 0)
          )
        ),
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "images gen / vision"),
          e(
            "div",
            { className: "dash-kpi-value" },
            fmtNum(summary && summary.totals ? summary.totals.images_generated : 0) +
              " / " +
              fmtNum(summary && summary.totals ? summary.totals.images_analyzed : 0)
          )
        ),
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "users (stats file)"),
          e("div", { className: "dash-kpi-value" }, fmtNum(summary && summary.totals ? summary.totals.users : 0))
        ),
        e(
          "div",
          { className: "dash-kpi" },
          e("div", { className: "dash-kpi-label" }, "bot chat responses"),
          e("div", { className: "dash-kpi-value" }, fmtNum(summary && summary.totals ? summary.totals.chats : 0))
        )
      ),
      e(
        "div",
        { className: "stats-chart-grid" },
        e(
          "article",
          { className: "stats-chart-panel" },
          e("h4", { style: { marginTop: 0 } }, "Last 7 days — normalized overlay"),
          e(
            "div",
            { className: "stats-toggles" },
            e(
              "label",
              null,
              e("input", {
                type: "checkbox",
                checked: seriesMask.messages,
                onChange: function () {
                  toggleMask("messages");
                },
              }),
              " All messages"
            ),
            e(
              "label",
              null,
              e("input", {
                type: "checkbox",
                checked: seriesMask.gen,
                onChange: function () {
                  toggleMask("gen");
                },
              }),
              " Generation"
            ),
            e(
              "label",
              null,
              e("input", {
                type: "checkbox",
                checked: seriesMask.vis,
                onChange: function () {
                  toggleMask("vis");
                },
              }),
              " Vision"
            ),
            e(
              "label",
              null,
              e("input", {
                type: "checkbox",
                checked: seriesMask.other,
                onChange: function () {
                  toggleMask("other");
                },
              }),
              " Other chat"
            )
          ),
          e(NormalizedMultiLine, {
            labels: dayLabels,
            series: multiSeries,
            title: null,
          })
        ),
        e(
          "article",
          { className: "stats-chart-panel" },
          e(
            "h4",
            { style: { marginTop: 0 } },
            fromSqliteArchive ? "Rolling 24h — archive message volume" : "Rolling 24h — stacked volume"
          ),
          e(HourlyStackedBars, {
            labels: hourly.labels,
            totals: hourly.messages,
            generated: hourly.images_generated,
            analyzed: hourly.images_analyzed,
            title: null,
            archiveVolumeOnly: fromSqliteArchive,
          })
        )
      ),
      e(
        "div",
        { className: "stats-spark-grid" },
        e(SparkLine, {
          title: "7d — all messages",
          points: s7.messages || [],
          labels: dayLabels,
          color: "#5fa8ff",
          compact: true,
        }),
        e(SparkLine, {
          title: "7d — image generation",
          points: s7.images_generated || [],
          labels: dayLabels,
          color: "#9ece6a",
          compact: true,
        }),
        e(SparkLine, {
          title: "7d — vision",
          points: s7.images_analyzed || [],
          labels: dayLabels,
          color: "#bb9af7",
          compact: true,
        }),
        e(SparkLine, {
          title: "7d — other messages",
          points: s7.other_messages || [],
          labels: dayLabels,
          color: "#e0af68",
          compact: true,
        })
      ),
      e(
        "div",
        { className: "stats-event-section" },
        e("h4", { style: { marginTop: 0 } }, "Event mix (all time, from archive)"),
        !eventPairs.length
          ? e("p", { className: "muted" }, "No event_type data yet.")
          : e(
              "div",
              { className: "stats-event-chips" },
              eventPairs.map(function (x, i) {
                return e(
                  "div",
                  { className: "dash-kpi stats-event-chip", key: i },
                  e("div", { className: "dash-kpi-label mono" }, x.k),
                  e("div", { className: "dash-kpi-value" }, fmtNum(x.n))
                );
              })
            )
      ),
      e(
        "div",
        { className: "stats-leaders-grid" },
        e(LeaderBoard, {
          title: "Top users — messages",
          rows: (summary && summary.top && summary.top.users_by_messages) || [],
          valueKey: "messages",
          labelKey: "username",
        }),
        e(LeaderBoard, {
          title: "Top users — images generated",
          rows: (summary && summary.top && summary.top.users_by_images) || [],
          valueKey: "images_generated",
          labelKey: "username",
        }),
        e(LeaderBoard, {
          title: "Top users — vision events",
          rows: (summary && summary.top && summary.top.users_by_vision) || [],
          valueKey: "vision",
          labelKey: "username",
        }),
        e(LeaderBoard, {
          title: "Top channels — messages",
          rows: (summary && summary.top && summary.top.channels_by_messages) || [],
          valueKey: "messages",
          labelKey: "channel_name",
          fallback: "channel",
        }),
        e(LeaderBoard, {
          title: "Top channels — images",
          rows: (summary && summary.top && summary.top.channels_by_images) || [],
          valueKey: "images_generated",
          labelKey: "channel_name",
          fallback: "channel",
        })
      )
    );
  }

  async function fetchArchiveMessageByImage(filename) {
    try {
      var res = await fetch("/api/archive/message_by_image?filename=" + encodeURIComponent(filename));
      var data = await res.json();
      if (!data || !data.ok) return null;
      return data;
    } catch (_e) {
      return null;
    }
  }

  function MediaArchiveTab({ active }) {
    const [imgItems, setImgItems] = useState([]);
    const [imgTotal, setImgTotal] = useState(0);
    const [imgOffset, setImgOffset] = useState(0);
    const imgLimit = 30;
    const [imgKind, setImgKind] = useState("all");
    const [msgItems, setMsgItems] = useState([]);
    const [msgTotal, setMsgTotal] = useState(0);
    const [msgOffset, setMsgOffset] = useState(0);
    const msgLimit = 40;
    const [detail, setDetail] = useState(null);

    const loadImages = useCallback(
      async function (off) {
        try {
          var qs = "/api/archive/images?limit=" + imgLimit + "&offset=" + (off || 0);
          if (imgKind && imgKind !== "all") qs += "&kind=" + encodeURIComponent(imgKind);
          var data = await readJson(qs);
          setImgItems(data.items || []);
          setImgTotal(Number(data.total || 0));
          setImgOffset(off || 0);
        } catch (_err) {
          setImgItems([]);
          setImgTotal(0);
        }
      },
      [imgKind, imgLimit]
    );

    const loadMessages = useCallback(
      async function (off) {
        try {
          var data = await readJson("/api/archive/messages?limit=" + msgLimit + "&offset=" + (off || 0));
          setMsgItems(data.items || []);
          setMsgTotal(Number(data.total || 0));
          setMsgOffset(off || 0);
        } catch (_err) {
          setMsgItems([]);
          setMsgTotal(0);
        }
      },
      [msgLimit]
    );

    useEffect(
      function () {
        if (!active) return;
        loadMessages(0);
      },
      [active, loadMessages]
    );

    useEffect(
      function () {
        if (!active) return;
        loadImages(0);
        setDetail(null);
      },
      [active, imgKind, loadImages]
    );

    useEffect(
      function () {
        if (!active) return;
        function onKey(ev) {
          if (ev.key === "Escape") setDetail(null);
        }
        window.addEventListener("keydown", onKey);
        return function () {
          window.removeEventListener("keydown", onKey);
        };
      },
      [active]
    );

    async function selectImage(row) {
      var fn = row.filename;
      if (!fn) return;
      var isVis = row.event_type === "vision";
      if (isVis) {
        setDetail({ loading: true, row: row, filename: fn });
        var d = await fetchArchiveMessageByImage(fn);
        if (d) {
          setDetail({
            loading: false,
            row: row,
            filename: fn,
            username: d.username || row.username,
            ts: d.ts || row.ts,
            event_type: d.event_type || "vision",
            content: d.content || "",
          });
        } else {
          setDetail({
            loading: false,
            row: row,
            filename: fn,
            username: row.username,
            ts: row.ts,
            event_type: "vision",
            content: row.prompt || "(No matching messages.jsonl entry — showing index prompt only.)",
            fromIndexOnly: true,
          });
        }
      } else {
        setDetail({
          loading: false,
          row: row,
          filename: fn,
          username: row.username,
          ts: row.ts,
          event_type: "generated",
          content: row.prompt || "",
          meta: { w: row.width, h: row.height, seed: row.seed },
        });
      }
    }

    function rowSelected(row) {
      return !!(detail && detail.row && detail.row.filename === row.filename);
    }

    return e(
      "div",
      { className: "media-archive" },
      e(
        "p",
        { className: "muted", style: { marginTop: 0 } },
        "Light table: thumbnails only — click for caption (user, description, metadata). Live stream: Overview → Live log."
      ),
      e(
        "article",
        { className: "archive-panel archive-lighttable" },
        e("h4", { style: { marginTop: 0 } }, "Image archive"),
        e(
          "div",
          { className: "dash-controls", style: { flexWrap: "wrap", gap: "0.5rem", alignItems: "center", marginBottom: "0.65rem" } },
            e(
              "label",
              { style: { display: "flex", alignItems: "center", gap: "0.35rem" } },
              "Show",
              e(
                "select",
                {
                  value: imgKind,
                  onChange: function (ev) {
                    setImgKind(ev.target.value || "all");
                  },
                },
                e("option", { value: "all" }, "All"),
                e("option", { value: "vision" }, "Vision / analyzed"),
                e("option", { value: "generated" }, "Generated (SD)")
              )
            ),
            e(
              "button",
              { type: "button", className: "secondary", disabled: imgOffset <= 0, onClick: function () { loadImages(Math.max(0, imgOffset - imgLimit)); } },
              "Prev page"
            ),
            e(
              "button",
              {
                type: "button",
                className: "secondary",
                disabled: imgOffset + imgLimit >= imgTotal,
                onClick: function () {
                  if (imgOffset + imgLimit < imgTotal) loadImages(imgOffset + imgLimit);
                },
              },
              "Next page"
            ),
            e(
              "span",
              { className: "muted", style: { fontSize: "0.88rem" } },
              fmtNum(imgTotal ? imgOffset + 1 : 0) +
                "–" +
                fmtNum(Math.min(imgOffset + imgLimit, imgTotal)) +
                " of " +
                fmtNum(imgTotal)
            )
        ),
        e(
          "div",
          { className: "archive-img-grid archive-img-grid--light" },
          !imgItems.length
            ? e("p", { className: "muted" }, "No images in the index yet.")
            : imgItems.map(function (row, i) {
                var fn = row.filename || "";
                var isVis = row.event_type === "vision";
                var sel = rowSelected(row);
                return e(
                  "div",
                  {
                    key: i,
                    className: "archive-img-card" + (sel ? " archive-img-card--selected" : ""),
                    role: "button",
                    tabIndex: 0,
                    "aria-pressed": sel,
                    onClick: function () {
                      selectImage(row);
                    },
                    onKeyDown: function (ev) {
                      if (ev.key === "Enter" || ev.key === " ") {
                        ev.preventDefault();
                        selectImage(row);
                      }
                    },
                  },
                  e(
                    "div",
                    { className: "archive-img-thumb-wrap" },
                    e("img", {
                      className: "archive-img-thumb",
                      alt: "",
                      loading: "lazy",
                      draggable: false,
                      src: "/media/thumbs/" + encodeURIComponent(fn),
                      onError: function (ev) {
                        ev.target.src = "/media/images/" + encodeURIComponent(fn);
                      },
                    }),
                    isVis
                      ? e("span", { className: "archive-img-badge" }, "V")
                      : e("span", { className: "archive-img-badge archive-img-badge--gen" }, "SD")
                  )
                );
              })
        ),
        e(
          "div",
          { className: "archive-caption" },
          e("div", { className: "archive-caption-bar" }, e("span", { className: "archive-caption-label" }, "Caption")),
          !detail
            ? e("p", { className: "muted archive-caption-placeholder" }, "Select an image above. Press Esc to clear.")
            : detail.loading
              ? e("p", { className: "muted archive-caption-placeholder" }, "Loading vision text from message log…")
              : e(
                  "div",
                  { className: "archive-caption-inner" },
                  e(
                    "div",
                    { className: "archive-caption-visual" },
                    e("img", {
                      className: "archive-caption-img",
                      alt: "",
                      src: "/media/images/" + encodeURIComponent(detail.filename || ""),
                      onError: function (ev) {
                        ev.target.src = "/media/thumbs/" + encodeURIComponent(detail.filename || "");
                      },
                    })
                  ),
                  e(
                    "div",
                    { className: "archive-caption-body" },
                    e(
                      "div",
                      { className: "archive-caption-meta" },
                      e("span", { className: "pill" }, detail.event_type === "vision" ? "Vision" : "Generated"),
                      e("strong", { className: "archive-caption-user" }, detail.username || "—"),
                      e("span", { className: "muted mono archive-caption-ts" }, (detail.ts || "").replace("T", " ").slice(0, 19)),
                      detail.meta &&
                        detail.meta.w &&
                        detail.meta.h &&
                        e(
                          "span",
                          { className: "muted mono" },
                          detail.meta.w + "×" + detail.meta.h + (detail.meta.seed != null ? " · seed " + detail.meta.seed : "")
                        ),
                      detail.row &&
                        detail.row.channel_id != null &&
                        e("span", { className: "muted mono" }, "ch " + detail.row.channel_id),
                      detail.row &&
                        detail.row.original_url &&
                        e(
                          "a",
                          {
                            className: "archive-caption-src",
                            href: detail.row.original_url,
                            target: "_blank",
                            rel: "noreferrer",
                          },
                          "Source URL"
                        )
                    ),
                    e("div", { className: "mono archive-caption-filename" }, detail.filename),
                    detail.fromIndexOnly
                      ? e("p", { className: "muted archive-caption-note" }, "Full vision text not in messages.jsonl — index snippet only.")
                      : null,
                    e("div", { className: "archive-caption-text" }, detail.content || e("span", { className: "muted" }, "(no description)")),
                    e(
                      "div",
                      { className: "archive-caption-actions" },
                      e("button", { type: "button", className: "secondary", onClick: function () { setDetail(null); } }, "Clear selection"),
                      e(
                        "a",
                        {
                          role: "button",
                          href: "/media/images/" + encodeURIComponent(detail.filename || ""),
                          target: "_blank",
                          rel: "noreferrer",
                        },
                        "Open full file"
                      )
                    )
                  )
                )
        )
      ),
      e(
        "details",
        { className: "archive-details-log" },
        e("summary", null, "Text activity log (messages.jsonl)"),
        e(
          "div",
          { className: "archive-panel", style: { marginTop: "0.65rem" } },
          e(
            "div",
            { className: "dash-controls", style: { flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.65rem" } },
            e(
              "button",
              { type: "button", className: "secondary", disabled: msgOffset <= 0, onClick: function () { loadMessages(Math.max(0, msgOffset - msgLimit)); } },
              "Newer"
            ),
            e(
              "button",
              {
                type: "button",
                className: "secondary",
                disabled: msgOffset + msgLimit >= msgTotal,
                onClick: function () {
                  if (msgOffset + msgLimit < msgTotal) loadMessages(msgOffset + msgLimit);
                },
              },
              "Older"
            ),
            e(
              "span",
              { className: "muted", style: { fontSize: "0.88rem" } },
              fmtNum(msgTotal ? msgOffset + 1 : 0) +
                "–" +
                fmtNum(Math.min(msgOffset + msgLimit, msgTotal)) +
                " of " +
                fmtNum(msgTotal)
            )
          ),
          e(
            "div",
            { className: "archive-log-list" },
            !msgItems.length
              ? e("p", { className: "muted" }, "No log lines yet.")
              : msgItems.map(function (row, i) {
                  var et = row.event_type || "message";
                  var short = String(row.content || "").replace(/\s+/g, " ").trim();
                  if (short.length > 280) short = short.slice(0, 277) + "…";
                  return e(
                    "div",
                    { key: i, className: "archive-log-row" },
                    e("div", { className: "archive-log-meta mono" }, (row.ts || "").replace("T", " ").slice(0, 19)),
                    e("div", { className: "archive-log-who" }, row.username || "—"),
                    e("span", { className: "pill archive-et" }, et),
                    row.image_filename
                      ? e(
                          "a",
                          {
                            className: "archive-img-link",
                            href: "/media/images/" + encodeURIComponent(row.image_filename),
                            target: "_blank",
                            rel: "noreferrer",
                          },
                          row.image_filename
                        )
                      : null,
                    e("div", { className: "archive-log-body" }, short || e("span", { className: "muted" }, "(empty)"))
                  );
                })
          )
        )
      )
    );
  }

  function App() {
    const initialDataNode = document.getElementById("dashboard-initial");
    let initialStatus = { running: false, pid: null };
    try {
      if (initialDataNode) initialStatus = JSON.parse(initialDataNode.textContent || "{}").status || initialStatus;
    } catch (_err) {}

    const [tab, setTab] = useState("overview");
    const [notice, setNotice] = useState("");
    const [noticeError, setNoticeError] = useState(false);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState(initialStatus);
    const [summary, setSummary] = useState(null);
    const [envVars, setEnvVars] = useState({});
    const [selectedModel, setSelectedModel] = useState("");
    const [modelQuery, setModelQuery] = useState("");
    const [modelCatalogDraft, setModelCatalogDraft] = useState("");
    const [contextWindowSize, setContextWindowSize] = useState("16000");
    const [preset, setPreset] = useState("BEHAVIOUR");
    const [personalityText, setPersonalityText] = useState("");
    const [selectedCommand, setSelectedCommand] = useState(null);
    const [cmdEnvDrafts, setCmdEnvDrafts] = useState({});
    const [dashStatus, setDashStatus] = useState({});
    const [activity, setActivity] = useState({});
    const [cmdChannelList, setCmdChannelList] = useState([]);
    const [cmdChannelConfig, setCmdChannelConfig] = useState({});
    const [logs, setLogs] = useState(["[loading logs...]"]);
    const [autoscroll, setAutoscroll] = useState(true);
    const [pauseLogs, setPauseLogs] = useState(false);
    const [dbStatus, setDbStatus] = useState({ databases: [] });
    const [dbGuild, setDbGuild] = useState("");
    const [dbTable, setDbTable] = useState("messages");
    const [dbSearch, setDbSearch] = useState("");
    const [dbChannel, setDbChannel] = useState("");
    const [dbUser, setDbUser] = useState("");
    const [dbLimit, setDbLimit] = useState(50);
    const [dbHasImages, setDbHasImages] = useState(false);
    const [dbHasUrls, setDbHasUrls] = useState(false);
    const [dbOffset, setDbOffset] = useState(0);
    const [dbRows, setDbRows] = useState([]);
    const [dbTotal, setDbTotal] = useState(0);
    const [dbInfo, setDbInfo] = useState("Choose a server and click Load Rows.");
    const [dbSelectedRow, setDbSelectedRow] = useState(null);
    const [profSearch, setProfSearch] = useState("");
    const [profLimit, setProfLimit] = useState(50);
    const [profOffset, setProfOffset] = useState(0);
    const [profRows, setProfRows] = useState([]);
    const [profTotal, setProfTotal] = useState(0);
    const [profInfo, setProfInfo] = useState("Choose a server and click Load profiles.");
    const [profSelected, setProfSelected] = useState(null);
    const [dbPickerUsers, setDbPickerUsers] = useState([]);
    const [profPickerUsers, setProfPickerUsers] = useState([]);
    const [profileRowCount, setProfileRowCount] = useState(null);
    const [userPickerError, setUserPickerError] = useState("");
    const [userPickerLoading, setUserPickerLoading] = useState(false);
    const [profBatchStatus, setProfBatchStatus] = useState(null);
    const [profBatchLogLines, setProfBatchLogLines] = useState([]);
    const [profBatchPollEpoch, setProfBatchPollEpoch] = useState(0);
    const [runtimeFlags, setRuntimeFlags] = useState({ rag_enabled: false });
    const [ragStatus, setRagStatus] = useState(null);
    const [profileStatus, setProfileStatus] = useState(null);
    const [archiveScanMinutes, setArchiveScanMinutes] = useState("0");
    // Overview tab — sticky-bar restart confirm + collapsible "knob" sections
    const [restartConfirmOpen, setRestartConfirmOpen] = useState(false);
    const [overviewKnobs, setOverviewKnobs] = useState({ commands: false, rag: false });
    const logRef = useRef(null);
    const profBatchLogRef = useRef(null);
    /** Last "next_index:profile_row_count" — refresh when either changes */
    const profBatchPickerSyncRef = useRef(null);

    const models = useMemo(function () {
      const list = parseModelList(envVars.AVAILABLE_MODELS || "");
      const current = String(envVars.LOCAL_CHAT || "");
      const normalized = uniqueModels([current].concat(list));
      return normalized.sort(function (a, b) {
        if (a === current) return -1;
        if (b === current) return 1;
        return a.toLowerCase().localeCompare(b.toLowerCase());
      });
    }, [envVars]);

    const filteredModels = useMemo(
      function () {
        const q = modelQuery.trim().toLowerCase();
        if (!q) return models;
        return models.filter(function (m) {
          return m.toLowerCase().indexOf(q) !== -1;
        });
      },
      [models, modelQuery]
    );

    const personalityOptions = useMemo(
      function () {
        // Only two prompts: main BEHAVIOUR and BEHAVIOUR_SEARCH
        var opts = [
          { key: "BEHAVIOUR", label: "Main chat personality", hint: "BEHAVIOUR" },
          { key: "BEHAVIOUR_SEARCH", label: "Search personality", hint: "BEHAVIOUR_SEARCH" },
        ];
        return opts;
      },
      [envVars]
    );

    useEffect(function () {
      loadSummary();
      loadEnv();
      loadRuntimeFlags();
      refreshStatus();
      loadEnvToggles();
      loadActivity();
      connectLogs();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(
      function () {
        if (tab === "database" || tab === "overview") {
          loadDatabases();
        }
        if (tab === "stats") {
          loadSummary();
        }
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [tab]
    );

    async function loadUserPickerData() {
      if (!dbGuild) {
        setDbPickerUsers([]);
        setProfPickerUsers([]);
        setProfileRowCount(null);
        setUserPickerError("");
        return;
      }
      setUserPickerLoading(true);
      setUserPickerError("");
      setProfileRowCount(null);
      try {
        const data = await readJson("/api/database/user-picker/" + encodeURIComponent(dbGuild));
        setDbPickerUsers(data.from_messages || []);
        setProfPickerUsers(data.from_profiles || []);
        if (typeof data.profile_row_count === "number") {
          setProfileRowCount(data.profile_row_count);
        } else {
          setProfileRowCount((data.from_profiles || []).length);
        }
        if ((data.from_profiles || []).length === 0 && (data.profile_row_count || 0) > 0) {
          setUserPickerError(
            "Profile rows exist in the database but the list came back empty — try Refresh lists or restart the web app."
          );
        }
      } catch (err) {
        setDbPickerUsers([]);
        setProfPickerUsers([]);
        setProfileRowCount(null);
        setUserPickerError(err.message || "Failed to load user lists");
      } finally {
        setUserPickerLoading(false);
      }
    }

    useEffect(
      function () {
        if (tab !== "database" || !dbGuild) return;
        loadUserPickerData();
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [tab, dbGuild]
    );

    useEffect(
      function () {
        if (tab !== "database" || !dbGuild) {
          setProfBatchStatus(null);
          setProfBatchLogLines([]);
          profBatchPickerSyncRef.current = null;
          return;
        }
        profBatchPickerSyncRef.current = null;
        var cancelled = false;
        var timeoutId = null;
        var BATCH_POLL_ACTIVE_MS = 12000;
        var BATCH_POLL_IDLE_MS = 120000;
        var BATCH_POLL_ERR_MS = 30000;
        function needActivePoll(st) {
          if (!st || !st.ok) return false;
          if (st.task_running) return true;
          var s = String(st.status || "");
          return s === "running" || s === "paused";
        }
        function schedule(delayMs) {
          if (cancelled) return;
          timeoutId = setTimeout(tick, delayMs);
        }
        async function tick() {
          if (cancelled) return;
          var nextDelay = BATCH_POLL_IDLE_MS;
          try {
            const d = await readJson("/api/profiles/batch/status/" + encodeURIComponent(dbGuild));
            if (!cancelled) {
              setProfBatchStatus(d);
              setProfBatchLogLines(Array.isArray(d.log_lines) ? d.log_lines : []);
              var ni = d.next_index != null ? Number(d.next_index) : 0;
              var prc =
                typeof d.profile_row_count === "number" && !Number.isNaN(d.profile_row_count)
                  ? Number(d.profile_row_count)
                  : null;
              var syncKey = prc !== null ? ni + ":" + prc : String(ni);
              if (syncKey !== profBatchPickerSyncRef.current) {
                profBatchPickerSyncRef.current = syncKey;
                loadUserPickerData();
              }
              nextDelay = needActivePoll(d) ? BATCH_POLL_ACTIVE_MS : BATCH_POLL_IDLE_MS;
            }
          } catch (_err) {
            if (!cancelled) setProfBatchStatus(null);
            nextDelay = BATCH_POLL_ERR_MS;
          }
          if (!cancelled) schedule(nextDelay);
        }
        tick();
        return function () {
          cancelled = true;
          profBatchPickerSyncRef.current = null;
          if (timeoutId != null) clearTimeout(timeoutId);
        };
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [tab, dbGuild, profBatchPollEpoch]
    );

    useEffect(
      function () {
        if (tab !== "database") return;
        var el = profBatchLogRef.current;
        if (!el) return;
        el.scrollTop = el.scrollHeight;
      },
      [tab, profBatchLogLines]
    );

    useEffect(
      function () {
        if (tab !== "overview" || !dbGuild) {
          return;
        }
        let cancelled = false;
        readJson("/api/rag/status/" + encodeURIComponent(dbGuild))
          .then(function (data) {
            if (!cancelled) setRagStatus(data);
          })
          .catch(function () {
            if (!cancelled) setRagStatus(null);
          });
        readJson("/api/profiles/status/" + encodeURIComponent(dbGuild))
          .then(function (data) {
            if (!cancelled) setProfileStatus(data);
          })
          .catch(function () {
            if (!cancelled) setProfileStatus(null);
          });
        return function () {
          cancelled = true;
        };
      },
      [tab, dbGuild]
    );

    useEffect(
      function () {
        if (!dbGuild || !(dbStatus.databases || []).length) {
          setArchiveScanMinutes("0");
          return;
        }
        var d = (dbStatus.databases || []).find(function (x) {
          return String(x.guild_id) === String(dbGuild);
        });
        if (d && d.archive_scan_interval_minutes != null) {
          setArchiveScanMinutes(String(d.archive_scan_interval_minutes));
        } else {
          setArchiveScanMinutes("0");
        }
      },
      [dbGuild, dbStatus]
    );

    useEffect(
      function () {
        if (autoscroll && logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
      },
      [logs, autoscroll]
    );

    const loadSummary = useCallback(async function () {
      try {
        const data = await readJson("/api/stats/summary?limit=12");
        setSummary(data);
      } catch (err) {
        setNotice("Failed to load stats: " + err.message);
        setNoticeError(true);
      }
    }, []);

    async function refreshStatus() {
      try {
        const data = await readJson("/api/bot/status");
        setStatus(data);
      } catch (err) {
        setNotice("Failed to refresh status: " + err.message);
        setNoticeError(true);
      }
      // Also load extended dashboard status
      try {
        const ds = await readJson("/api/bot/dashboard-status");
        console.log("dashboard-status response:", ds);
        setDashStatus(ds || {});
      } catch (dsErr) {
        console.error("dashboard-status fetch failed:", dsErr);
      }
    }

    // Track env var state for toggles (read directly from .env-stable)
    const [envToggles, setEnvToggles] = useState({});

    async function loadEnvToggles() {
      try {
        var data = await readJson("/api/env/get");
        var vars = data.vars || {};
        setEnvToggles({
          DAILY_POST_ENABLED: (vars.DAILY_POST_ENABLED || "").toLowerCase() === "true",
          BLUESKY_AUTO_REPLY: (vars.BLUESKY_AUTO_REPLY || "").toLowerCase() === "true",
        });
      } catch (err) {
        console.error("Failed to load env toggles:", err);
      }
    }

    async function toggleEnvVar(key) {
      var current = envToggles[key];
      var newVal = current ? "false" : "true";
      try {
        var payload = {};
        payload[key] = newVal;
        await fetch("/api/env/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        // Update local state immediately for responsive UI
        var updated = Object.assign({}, envToggles);
        updated[key] = !current;
        setEnvToggles(updated);
      } catch (err) {
        console.error("Failed to toggle " + key + ":", err);
      }
    }

    async function loadActivity() {
      try {
        var data = await readJson("/api/bot/activity");
        setActivity(data || {});
      } catch (err) {
        console.error("Failed to load activity:", err);
      }
    }

    async function botAction(path, successMsg) {
      setBusy(true);
      setNotice("Working...");
      setNoticeError(false);
      try {
        const data = await readJson(path, { method: "POST" });
        setStatus(data);
        if (successMsg) setNotice(successMsg);
        else setNotice(data.message || "Done");
      } catch (err) {
        setNotice("Action failed: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    function connectLogs() {
      try {
        const proto = window.location.protocol === "https:" ? "wss" : "ws";
        const ws = new WebSocket(proto + "://" + window.location.host + "/ws/logs");
        ws.onopen = function () {
          pushLog("[logs connected]");
        };
        ws.onmessage = function (ev) {
          if (!pauseLogs) pushLog(String(ev.data || ""));
        };
        ws.onclose = function () {
          pushLog("[logs disconnected]");
        };
      } catch (_err) {
        pushLog("[logs unavailable]");
      }
    }

    function pushLog(line) {
      setLogs(function (curr) {
        const next = curr.concat([line]);
        if (next.length > 900) return next.slice(next.length - 700);
        return next;
      });
    }

    async function loadRuntimeFlags() {
      try {
        const data = await readJson("/api/runtime-flags");
        setRuntimeFlags(data || { rag_enabled: false });
      } catch (_err) {
        setRuntimeFlags({ rag_enabled: false });
      }
    }

    async function rebuildRagIndex() {
      if (!dbGuild) return;
      if (
        !window.confirm(
          "Rebuild RAG index for this server? This calls LM Studio embeddings and may take a while."
        )
      )
        return;
      setBusy(true);
      setNotice("Rebuilding RAG index…");
      setNoticeError(false);
      try {
        const data = await readJson("/api/rag/reindex/" + encodeURIComponent(dbGuild), { method: "POST" });
        setNotice("RAG index rebuilt: " + fmtNum(data.chunks || 0) + " chunks.");
        const st = await readJson("/api/rag/status/" + encodeURIComponent(dbGuild));
        setRagStatus(st);
      } catch (err) {
        setNotice("RAG reindex failed: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function profileResetAll() {
      if (!dbGuild) return;
      if (
        !window.confirm(
          "Delete ALL stored member profiles for this server and clear batch job state? " +
            "You can run a new batch afterward."
        )
      )
        return;
      setBusy(true);
      setNoticeError(false);
      try {
        await readJson("/api/profiles/reset/" + encodeURIComponent(dbGuild), { method: "POST" });
        setNotice("All profiles deleted for this server.");
        setProfBatchPollEpoch(function (n) {
          return n + 1;
        });
        loadUserPickerData();
        loadProfileList(0);
        loadDatabases();
        try {
          const st = await readJson("/api/profiles/status/" + encodeURIComponent(dbGuild));
          setProfileStatus(st);
        } catch (_err) {
          setProfileStatus(null);
        }
      } catch (err) {
        setNotice("Reset failed: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function profileBatchStart() {
      if (!dbGuild) return;
      if (
        !window.confirm(
          "Start profile batch? One LLM call per user (LOCAL_CHAT). You can pause, resume, and cancel. Logs appear below."
        )
      )
        return;
      setBusy(true);
      setNoticeError(false);
      try {
        const data = await readJson("/api/profiles/batch/start/" + encodeURIComponent(dbGuild), { method: "POST" });
        setNotice("Batch queued: " + fmtNum(data.total || 0) + " user(s).");
        setProfBatchPollEpoch(function (n) {
          return n + 1;
        });
      } catch (err) {
        setNotice("Batch start failed: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function profileBatchPause() {
      if (!dbGuild) return;
      try {
        await readJson("/api/profiles/batch/pause/" + encodeURIComponent(dbGuild), { method: "POST" });
        setProfBatchPollEpoch(function (n) {
          return n + 1;
        });
      } catch (_err) {}
    }

    async function profileBatchResume() {
      if (!dbGuild) return;
      try {
        await readJson("/api/profiles/batch/resume/" + encodeURIComponent(dbGuild), { method: "POST" });
        setProfBatchPollEpoch(function (n) {
          return n + 1;
        });
      } catch (_err) {}
    }

    async function profileBatchCancel() {
      if (!dbGuild) return;
      if (!window.confirm("Cancel the profile worker? Current user may still finish.")) return;
      try {
        await readJson("/api/profiles/batch/cancel/" + encodeURIComponent(dbGuild), { method: "POST" });
        setProfBatchPollEpoch(function (n) {
          return n + 1;
        });
      } catch (_err) {}
    }

    async function loadEnv() {
      try {
        const data = await readJson("/api/env/get");
        const vars = data.vars || {};
        setEnvVars(vars);
        setSelectedModel(String(vars.LOCAL_CHAT || ""));
        setContextWindowSize(String(vars.CONTEXT_WINDOW_TOKENS || "16000"));
        setModelCatalogDraft(parseModelList(vars.AVAILABLE_MODELS || "").join("\n"));
      } catch (err) {
        setNotice("Failed to load environment: " + err.message);
        setNoticeError(true);
      }
    }

    useEffect(
      function () {
        if (!personalityOptions.length) {
          setPreset("");
          setPersonalityText("");
          return;
        }
        const hasPreset = personalityOptions.some(function (opt) {
          return opt.key === preset;
        });
        const resolved = hasPreset ? preset : personalityOptions[0].key;
        if (!hasPreset) {
          setPreset(resolved);
        }
        setPersonalityText(formatBehaviourForEditor(normalizeLineEndings(envVars[resolved] || "")));
      },
      [preset, envVars, personalityOptions]
    );

    function getPresetValueRaw(key) {
      return normalizeLineEndings(envVars[key] || "");
    }

    function getPresetValueFormatted(key) {
      return formatBehaviourForEditor(getPresetValueRaw(key));
    }

    async function saveEnvOnly(updates, msg) {
      setBusy(true);
      setNotice("Saving...");
      setNoticeError(false);
      try {
        const payload = Object.assign({}, envVars, updates || {});
        await readJson("/api/env/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        setNotice(msg || "Saved to .env");
        await loadEnv();
      } catch (err) {
        setNotice("Save failed: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function saveEnvAndRestart(updates, msg) {
      setBusy(true);
      setNotice("Saving and restarting...");
      setNoticeError(false);
      try {
        const payload = Object.assign({}, envVars, updates || {});
        await readJson("/api/env/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const restarted = await readJson("/api/bot/restart", { method: "POST" });
        setStatus(restarted);
        setNotice(msg || "Saved and restarted.");
        await loadEnv();
      } catch (err) {
        setNotice("Update failed: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function saveAvailableModels() {
      const modelList = uniqueModels(parseModelList(modelCatalogDraft));
      const joined = modelList.join(", ");
      setModelCatalogDraft(modelList.join("\n"));
      await saveEnvAndRestart({ AVAILABLE_MODELS: joined }, "Model catalog saved and bot restarted.");
      if (modelList.length && modelList.indexOf(selectedModel) === -1) {
        setSelectedModel(modelList[0]);
      }
    }

    async function fetchLMStudioModels() {
      setBusy(true);
      setNotice("");
      setNoticeError(false);
      try {
        var resp = await fetch("/api/lm-studio/models");
        var data = await resp.json();
        if (!data.ok) {
          setNotice("Could not fetch models: " + (data.message || "unknown error"));
          setNoticeError(true);
          return;
        }
        var fetched = data.models || [];
        if (!fetched.length) {
          setNotice("LM Studio returned no models. Is a model downloaded?");
          setNoticeError(true);
          return;
        }
        // Extract model keys — the API now returns objects with key/label/type/quant
        var keys = fetched.map(function (m) {
          return typeof m === "string" ? m : (m.key || "");
        }).filter(Boolean);
        // Store the full model info for display labels
        var labelMap = {};
        fetched.forEach(function (m) {
          if (typeof m === "object" && m.key) {
            labelMap[m.key] = m.label || m.key;
          }
        });
        window._lmModelLabels = labelMap;
        // Merge with existing catalog
        var existing = parseModelList(modelCatalogDraft);
        var merged = uniqueModels(keys.concat(existing));
        setModelCatalogDraft(merged.join("\n"));
        var llmCount = fetched.filter(function (m) { return typeof m === "object" && m.type === "llm"; }).length;
        var embCount = fetched.filter(function (m) { return typeof m === "object" && m.type === "embedding"; }).length;
        setNotice("Fetched " + keys.length + " model(s) from LM Studio (" + llmCount + " LLM, " + embCount + " embedding). Review and save.");
      } catch (err) {
        setNotice("Failed to reach LM Studio: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function loadDatabases() {
      try {
        const data = await readJson("/api/database/status");
        setDbStatus(data);
        if ((data.databases || []).length) {
          const first = String(data.databases[0].guild_id || "");
          setDbGuild(function (old) {
            return old || first;
          });
        } else {
          setDbInfo("No databases found yet. Run /soupyscan in Discord first.");
        }
      } catch (err) {
        setDbInfo("Failed to load databases: " + err.message);
      }
    }

    async function loadRows(nextOffset) {
      const offset = nextOffset == null ? dbOffset : nextOffset;
      if (!dbGuild) return;
      setDbInfo("Loading rows...");
      setDbSelectedRow(null);
      try {
        const qs = new URLSearchParams({
          table: dbTable,
          limit: String(dbLimit),
          offset: String(offset),
          has_images: String(!!dbHasImages),
          has_urls: String(!!dbHasUrls),
        });
        if (dbSearch.trim()) qs.set("search", dbSearch.trim());
        if (dbChannel.trim()) qs.set("channel_id", dbChannel.trim());
        if (dbUser.trim()) qs.set("user_id", dbUser.trim());
        const data = await readJson("/api/database/explore/" + encodeURIComponent(dbGuild) + "?" + qs.toString());
        const total = Number(data.total || 0);
        const start = total ? offset + 1 : 0;
        const end = Math.min(offset + Number(dbLimit), total);
        setDbRows(data.rows || []);
        setDbTotal(total);
        setDbOffset(offset);
        setDbInfo("Showing " + start + "-" + end + " of " + total + " rows");
      } catch (err) {
        setDbInfo("Failed to load rows: " + err.message);
      }
    }

    async function loadProfileList(nextOffset) {
      const offset = nextOffset == null ? profOffset : nextOffset;
      if (!dbGuild) return;
      setProfInfo("Loading profiles...");
      setProfSelected(null);
      try {
        const qs = new URLSearchParams({ limit: String(profLimit), offset: String(offset) });
        if (profSearch.trim()) qs.set("search", profSearch.trim());
        const data = await readJson("/api/profiles/list/" + encodeURIComponent(dbGuild) + "?" + qs.toString());
        const total = Number(data.total || 0);
        const start = total ? offset + 1 : 0;
        const end = Math.min(offset + Number(profLimit), total);
        setProfRows(data.rows || []);
        setProfTotal(total);
        setProfOffset(offset);
        setProfInfo("Showing " + start + "-" + end + " of " + total + " stored profiles");
        loadUserPickerData();
      } catch (err) {
        setProfInfo("Failed to load profiles: " + err.message);
        setProfRows([]);
        setProfTotal(0);
      }
    }

    async function triggerScan() {
      if (!dbGuild) return;
      if (!window.confirm("Run incremental scan now?")) return;
      try {
        const data = await readJson("/api/database/scan/" + encodeURIComponent(dbGuild), { method: "POST" });
        setNotice(data.message || "Scan triggered");
        setNoticeError(false);
        setTimeout(loadDatabases, 1200);
      } catch (err) {
        setNotice("Scan failed: " + err.message);
        setNoticeError(true);
      }
    }

    async function saveArchiveSchedule() {
      if (!dbGuild) return;
      var m = parseInt(String(archiveScanMinutes).trim(), 10);
      if (isNaN(m) || m < 0 || m > 10080) {
        setNotice("Auto-archive minutes must be a number from 0 to 10080 (7 days). Use 0 to disable.");
        setNoticeError(true);
        return;
      }
      try {
        const data = await readJson("/api/database/archive-schedule/" + encodeURIComponent(dbGuild), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ minutes: m }),
        });
        setNotice(
          data.archive_scan_interval_minutes === 0
            ? "Automatic archive scans disabled for this server."
            : "Saved: incremental archive every " +
                data.archive_scan_interval_minutes +
                " min (bot polls ~45s). RAG reindexes after each successful scan."
        );
        setNoticeError(false);
        setTimeout(loadDatabases, 400);
      } catch (err) {
        setNotice("Auto-archive save failed: " + err.message);
        setNoticeError(true);
      }
    }

    const tabButton = function (id, label) {
      return e(
        "button",
        {
          type: "button",
          className: tab === id ? "" : "secondary",
          onClick: function () {
            setTab(id);
          },
        },
        label
      );
    };

    return e(
      "section",
      { className: "dash-app" },
      e(
        "article",
        { className: "dash-hero" },
        e(
          "div",
          null,
          e("h2", { style: { marginBottom: "0.2rem" } }, "Soupy Control Studio"),
          e("p", { className: "dash-subtitle" }, "Faster controls, cleaner stats, and a playful database explorer.")
        ),
        e(
          "div",
          { className: "dash-nav" },
          tabButton("overview", "Overview"),
          tabButton("model", "Model & Personality"),
          tabButton("stats", "Stats Studio"),
          tabButton("archive", "Media & log"),
          tabButton("database", "Database Explorer"),
          e("a", { role: "button", className: "secondary", href: "/env" }, "Environment Editor")
        )
      ),
      e("div", { className: noticeError ? "dash-message error" : "dash-message" }, notice),
      tab === "overview" && (function () {
        function formatUptime(s) {
          if (s == null) return "—";
          var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
          return (d > 0 ? d + "d " : "") + h + "h " + m + "m";
        }
        function timeUntil(iso) {
          if (!iso) return "—";
          if (typeof iso === "string" && isNaN(Date.parse(iso))) return iso;
          try {
            var diff = Math.max(0, Math.floor((new Date(iso).getTime() - Date.now()) / 60000));
            if (diff < 1) return "imminent";
            if (diff < 60) return "in " + diff + "m";
            return "in " + Math.floor(diff / 60) + "h " + (diff % 60) + "m";
          } catch (_) { return "—"; }
        }
        function timeAgo(iso) {
          if (!iso) return "never";
          try {
            var ago = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
            if (ago < 1) return "just now";
            if (ago < 60) return ago + "m ago";
            return Math.floor(ago / 60) + "h " + (ago % 60) + "m ago";
          } catch (_) { return "—"; }
        }
        function togglePill(on, onClick, disabled) {
          return e("span", {
            className: "dash-toggle-pill",
            "data-on": on ? "1" : "0",
            "aria-disabled": disabled ? "true" : "false",
            role: "switch",
            "aria-checked": on ? "true" : "false",
            onClick: disabled ? undefined : onClick
          }, e("span", { className: "dash-toggle-knob" }));
        }
        function loopCard(opts) {
          var meta = [
            e("span", { key: "nl", className: "dash-loop-card-meta-label" }, "Next:"),
            e("span", { key: "nv" }, timeUntil(opts.nextRunIso)),
            e("span", { key: "ll", className: "dash-loop-card-meta-label" }, "Last:"),
            e("span", { key: "lv" }, timeAgo(opts.lastRunIso))
          ];
          if (opts.intervalLabel) {
            meta.push(e("span", { key: "il", className: "dash-loop-card-meta-label" }, "Cycle:"));
            meta.push(e("span", { key: "iv" }, opts.intervalLabel));
          }
          if (opts.todayLine) {
            meta.push(e("span", { key: "tl", className: "dash-loop-card-meta-label" }, "Today:"));
            meta.push(e("span", { key: "tv" }, opts.todayLine));
          }
          return e("div", { key: opts.key, className: "dash-loop-card" },
            e("div", { className: "dash-loop-card-header" },
              e("h5", { className: "dash-loop-card-title" }, opts.title),
              opts.on != null
                ? togglePill(!!opts.on, opts.onToggle, !opts.onToggle || busy)
                : e("span", { className: "muted", style: { fontSize: "0.78rem" } }, "always on")
            ),
            opts.toggleHint
              ? e("div", { className: "muted", style: { fontSize: "0.76rem" } }, opts.toggleHint)
              : null,
            e("div", { className: "dash-loop-card-meta" }, meta),
            e("div", { className: "dash-loop-card-actions" },
              e("button", {
                className: "secondary",
                disabled: !!opts.runNowDisabled,
                title: opts.runNowTitle || null,
                onClick: opts.runNowOnClick || null
              }, "Run now")
            )
          );
        }
        function knobToggle(key) {
          return function () {
            setOverviewKnobs(function (prev) {
              var n = Object.assign({}, prev);
              n[key] = !prev[key];
              return n;
            });
          };
        }

        var pendingTitle = "Manual trigger plumbing not yet implemented (planned).";

        // ---------- Band 0 — Sticky status bar ----------
        var stickyBar = e(
          "div",
          { className: "dash-sticky-bar", role: "status", "aria-label": "Bot status" },
          e("div", { className: "dash-sticky-segment" },
            e("span", { className: "status-dot " + (status.running ? "status-on" : "status-off") }),
            e("strong", null, status.running ? "Running" : "Stopped"),
            status.pid ? e("span", { className: "muted mono", style: { fontSize: "0.78rem" } }, "PID " + status.pid) : null
          ),
          dashStatus.model
            ? e("div", { className: "dash-sticky-segment" },
                e("span", { className: "muted" }, "Model"),
                e("span", { className: "mono", style: { fontSize: "0.85rem" } }, dashStatus.model)
              )
            : null,
          dashStatus.uptime_seconds != null
            ? e("div", { className: "dash-sticky-segment" },
                e("span", { className: "muted" }, "Uptime"),
                e("span", null, formatUptime(dashStatus.uptime_seconds))
              )
            : null,
          e("div", { className: "dash-sticky-segment" },
            e("span", { className: "status-dot " + (dashStatus.llm_online ? "status-on" : "status-off") }),
            e("span", null, "LLM")
          ),
          e("div", { className: "dash-sticky-segment" },
            e("span", { className: "status-dot " + (dashStatus.sd_online ? "status-on" : "status-off") }),
            e("span", null, "SD")
          ),
          e("div", { className: "dash-sticky-spacer" }),
          e("div", { className: "dash-sticky-actions" },
            e("button", {
              disabled: busy || status.running,
              onClick: function () { botAction("/api/bot/start"); }
            }, "Start"),
            e("button", {
              className: "secondary",
              disabled: busy || !status.running,
              onClick: function () { botAction("/api/bot/stop"); }
            }, "Stop"),
            !restartConfirmOpen
              ? e("button", {
                  disabled: busy,
                  onClick: function () { setRestartConfirmOpen(true); }
                }, "Restart")
              : e("span", { className: "dash-confirm" },
                  e("span", { className: "dash-confirm-prompt" }, "Restart bot?"),
                  e("button", {
                    className: "dash-confirm-yes",
                    disabled: busy,
                    onClick: function () { setRestartConfirmOpen(false); botAction("/api/bot/restart"); }
                  }, "Yes"),
                  e("button", {
                    className: "secondary",
                    onClick: function () { setRestartConfirmOpen(false); }
                  }, "Cancel")
                ),
            e("button", {
              className: "secondary",
              disabled: busy,
              onClick: function () { refreshStatus(); loadSummary(); loadEnvToggles(); loadActivity(); }
            }, "Refresh")
          )
        );

        // ---------- Band 1 — Live Activity + Quick Actions ----------
        var bandLive = e(
          "div",
          { className: "dash-band-live" },
          e("article", null,
            e("h4", { style: { marginTop: 0 } }, "Live Activity"),
            e("div", { className: "dash-controls", style: { marginBottom: "0.55rem" } },
              e("button", { className: "secondary", onClick: function () { setPauseLogs(!pauseLogs); } },
                pauseLogs ? "Resume stream" : "Pause stream"),
              e("button", { className: "secondary", onClick: function () { setAutoscroll(!autoscroll); } },
                autoscroll ? "Auto-scroll on" : "Auto-scroll off"),
              e("button", { className: "secondary", onClick: function () { setLogs([]); } }, "Clear")
            ),
            e("div", { className: "dash-log mono", ref: logRef }, logs.join("\n"))
          ),
          e("article", { className: "dash-quick-card" },
            e("h4", null, "Quick Actions"),
            e("p", { className: "muted", style: { marginTop: 0, fontSize: "0.85rem" } },
              "Manual triggers — backend wiring lands in workstream 2; buttons activate then."),
            e("div", { className: "dash-quick-grid" },
              [
                ["Post a daily article", "uses the daily-post pipeline"],
                ["Drop a musing", "thinks out loud in the musing channel"],
                ["Bluesky: reply", "find a post and reply once"],
                ["Bluesky: quote-post", "quote-share something interesting"],
                ["Bluesky: original post", "find an article and post about it"],
                ["Run archive scan", "scan recent messages now"]
              ].map(function (pair, idx) {
                return e("button", {
                  key: idx,
                  className: "secondary",
                  disabled: true,
                  title: pendingTitle
                }, pair[0],
                  e("span", { className: "dash-quick-sub" }, pair[1])
                );
              })
            )
          )
        );

        // ---------- Band 2 — Loops grid ----------
        var timers = dashStatus.timers || {};
        var bskyAct = activity.bluesky || {};
        var dailyPostsToday = (activity.daily_posts || {}).posts_today || 0;
        var bandLoops = e(
          "div",
          { className: "dash-loop-grid" },
          loopCard({
            key: "loop-daily-post",
            title: "Daily Post (Discord)",
            on: !!envToggles.DAILY_POST_ENABLED,
            onToggle: function () { toggleEnvVar("DAILY_POST_ENABLED"); },
            toggleHint: "Toggling writes .env-stable; takes effect on next bot restart.",
            nextRunIso: (timers.daily_post || {}).next_run,
            lastRunIso: (timers.daily_post || {}).last_run,
            intervalLabel: (timers.daily_post || {}).interval || null,
            todayLine: dailyPostsToday + " / 2 posts",
            runNowDisabled: true,
            runNowTitle: pendingTitle
          }),
          loopCard({
            key: "loop-bluesky",
            title: "Bluesky Engagement",
            on: !!envToggles.BLUESKY_AUTO_REPLY,
            onToggle: function () { toggleEnvVar("BLUESKY_AUTO_REPLY"); },
            toggleHint: "Toggling writes .env-stable; takes effect on next bot restart.",
            nextRunIso: (timers.bluesky || {}).next_run,
            lastRunIso: (timers.bluesky || {}).last_run,
            intervalLabel: (timers.bluesky || {}).interval || null,
            todayLine: (bskyAct.replies_today || 0) + " replies · "
                     + (bskyAct.posts_today || 0) + " posts · "
                     + (bskyAct.reposts_today || 0) + " reposts",
            runNowDisabled: true,
            runNowTitle: pendingTitle
          }),
          loopCard({
            key: "loop-archive-scan",
            title: "Archive Scan",
            on: null,
            nextRunIso: (timers.archive_scan || {}).next_run,
            lastRunIso: (timers.archive_scan || {}).last_run,
            intervalLabel: (timers.archive_scan || {}).interval || null,
            todayLine: null,
            runNowDisabled: true,
            runNowTitle: pendingTitle
          }),
          loopCard({
            key: "loop-rag-reindex",
            title: "RAG Reindex",
            on: null,
            nextRunIso: (timers.rag_reindex || {}).next_run,
            lastRunIso: (timers.rag_reindex || {}).last_run,
            intervalLabel: (timers.rag_reindex || {}).interval || null,
            todayLine: ragStatus && ragStatus.ok
              ? fmtNum(ragStatus.rag_chunks || 0) + " chunks indexed"
              : null,
            runNowDisabled: !dbGuild || !(dbStatus.databases || []).length || busy,
            runNowOnClick: rebuildRagIndex,
            runNowTitle: !dbGuild ? "Pick a guild in Knobs → RAG" : null
          }),
          loopCard({
            key: "loop-self-reflect",
            title: "Self-Reflection",
            on: !!envToggles.SELF_MD_ENABLED,
            onToggle: function () { toggleEnvVar("SELF_MD_ENABLED"); },
            toggleHint: "Toggling writes .env-stable; takes effect on next bot restart.",
            nextRunIso: (timers.self_reflect || {}).next_run,
            lastRunIso: (timers.self_reflect || {}).last_run,
            intervalLabel: (timers.self_reflect || {}).interval || null,
            todayLine: dashStatus.self_md_pending != null
              ? dashStatus.self_md_pending + " interactions pending"
              : null,
            runNowDisabled: true,
            runNowTitle: pendingTitle
          }),
          loopCard({
            key: "loop-musings",
            title: "Musings",
            on: !!envToggles.MUSING_ENABLED,
            onToggle: function () { toggleEnvVar("MUSING_ENABLED"); },
            toggleHint: "Toggling writes .env-stable; takes effect on next bot restart.",
            nextRunIso: (timers.musings || {}).next_run,
            lastRunIso: (timers.musings || {}).last_run,
            intervalLabel: (timers.musings || {}).interval || null,
            todayLine: null,
            runNowDisabled: true,
            runNowTitle: pendingTitle
          })
        );

        // ---------- Band 3 — Snapshot stats ----------
        var totals = (summary && summary.totals) || {};
        var bandSnapshot = e(
          "div",
          { className: "dash-snapshot-strip" },
          e("div", { className: "dash-snapshot-kpi" },
            e("span", { className: "dash-snapshot-kpi-label" }, "Messages"),
            e("span", { className: "dash-snapshot-kpi-value" }, fmtNum(totals.messages || 0))
          ),
          e("div", { className: "dash-snapshot-kpi" },
            e("span", { className: "dash-snapshot-kpi-label" }, "Images"),
            e("span", { className: "dash-snapshot-kpi-value" }, fmtNum(totals.images || 0))
          ),
          e("div", { className: "dash-snapshot-kpi" },
            e("span", { className: "dash-snapshot-kpi-label" }, "Users"),
            e("span", { className: "dash-snapshot-kpi-value" }, fmtNum(totals.users || 0))
          ),
          e("div", { className: "dash-snapshot-kpi" },
            e("span", { className: "dash-snapshot-kpi-label" }, "Messages · 24h"),
            e("span", { className: "dash-snapshot-kpi-value" }, fmtNum(summary ? summary.last24_messages || 0 : 0))
          ),
          e("div", { className: "dash-snapshot-kpi" },
            e("span", { className: "dash-snapshot-kpi-label" }, "Messages · 7d"),
            e("span", { className: "dash-snapshot-kpi-value" }, fmtNum(summary ? summary.last7d_messages || 0 : 0))
          ),
          e("div", { className: "dash-snapshot-kpi" },
            e("span", { className: "dash-snapshot-kpi-label" }, "Messages · 30d"),
            e("span", { className: "dash-snapshot-kpi-value" }, fmtNum(summary ? summary.last30d_messages || 0 : 0))
          ),
          e("div", { className: "dash-snapshot-link" },
            e("button", { className: "secondary", onClick: function () { setTab("stats"); } }, "Open Stats Studio →")
          )
        );

        // ---------- Band 4 — Knobs accordion ----------
        var ragKnobBody = e("div", null,
          e("p", { className: "muted", style: { marginTop: 0 } },
            "When enabled, every reply in a server uses the scanned message database for context (no trigger word). ",
            "Off by default."),
          e("label", { style: { display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" } },
            e("input", {
              type: "checkbox",
              checked: !!runtimeFlags.rag_enabled,
              disabled: busy,
              onChange: function (ev) {
                var v = !!ev.target.checked;
                setBusy(true);
                setNoticeError(false);
                readJson("/api/runtime-flags", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ rag_enabled: v })
                })
                  .then(function (data) {
                    setRuntimeFlags(data || { rag_enabled: v });
                    setNotice(v ? "RAG features enabled." : "RAG features disabled.");
                  })
                  .catch(function (err) { setNotice("Failed to save: " + err.message); setNoticeError(true); })
                  .finally(function () { setBusy(false); });
              }
            }),
            e("span", null, "Enable RAG (all server replies; WebUI master switch)")
          ),
          e("div", { className: "dash-field", style: { marginTop: "0.75rem" } },
            e("label", null, "Guild for index"),
            e("select", {
              value: dbGuild,
              onChange: function (ev) { setDbGuild(ev.target.value); }
            },
              (dbStatus.databases || []).length
                ? (dbStatus.databases || []).map(function (d) {
                    return e("option", { key: d.guild_id, value: d.guild_id }, (d.guild_name || "Guild") + " (" + d.guild_id + ")");
                  })
                : [e("option", { key: "_", value: "" }, "No databases — run /soupyscan first")]
            )
          ),
          ragStatus && ragStatus.ok
            ? e("p", { className: "muted", style: { marginBottom: "0.5rem" } },
                "Indexed chunks: ",
                e("span", { className: "mono" }, fmtNum(ragStatus.rag_chunks || 0)),
                " · Messages in DB: ",
                e("span", { className: "mono" }, fmtNum(ragStatus.total_messages || 0)))
            : e("p", { className: "muted" }, "Select a guild to see RAG index status."),
          profileStatus && profileStatus.ok
            ? e("p", { className: "muted", style: { marginBottom: "0.5rem" } },
                "Stored member profiles: ",
                e("span", { className: "mono" }, fmtNum(profileStatus.profile_count || 0)),
                profileStatus.latest_updated_at
                  ? e("span", null, " · Last profile update: ", e("span", { className: "mono" }, String(profileStatus.latest_updated_at)))
                  : null)
            : dbGuild && (dbStatus.databases || []).length
              ? e("p", { className: "muted", style: { marginBottom: "0.5rem" } }, "Member profiles: use Database tab → Profile batch.")
              : null,
          e("div", { className: "dash-controls", style: { flexWrap: "wrap", gap: "0.5rem" } },
            e("button", {
              className: "secondary",
              disabled: busy || !dbGuild || !(dbStatus.databases || []).length,
              onClick: rebuildRagIndex
            }, "Rebuild RAG index"),
            e("button", {
              className: "secondary",
              disabled: !dbGuild || !(dbStatus.databases || []).length,
              onClick: function () { setTab("database"); }
            }, "Database → profile batch")
          ),
          e("p", { className: "muted", style: { marginTop: "0.5rem", fontSize: "0.88rem" } },
            "Set ", e("span", { className: "mono" }, "RAG_EMBEDDING_MODEL"),
            " in ", e("span", { className: "mono" }, ".env-stable"),
            " to the LM Studio embedding model id. Load that embedding model in LM Studio; ",
            e("span", { className: "mono" }, "OPENAI_BASE_URL"),
            " should point at your server. Profile batch uses ",
            e("span", { className: "mono" }, "LOCAL_CHAT"), ", ",
            e("span", { className: "mono" }, "USER_PROFILES_BATCH_MAX_USERS"), ", ",
            e("span", { className: "mono" }, "USER_PROFILE_SAMPLE_MESSAGES"),
            " (see ", e("span", { className: "mono" }, ".env-stable"), ").")
        );

        // Slash commands knob body — wraps the original IIFE result
        var slashCommandsBody =           (function () {
            var COMMAND_CONFIG = {
              sd: {
                desc: "Generate images via Stable Diffusion",
                keys: ["SD_SERVER_URL", "SD_STEPS", "SD_GUIDANCE", "SD_NEGATIVE_PROMPT", "SD_DEFAULT_WIDTH", "SD_DEFAULT_HEIGHT", "SD_WIDE_WIDTH", "SD_WIDE_HEIGHT", "SD_TALL_WIDTH", "SD_TALL_HEIGHT"],
              },
              img2img: {
                desc: "Transform an image with a prompt",
                keys: ["SD_IMG2IMG_URL", "SD_NEGATIVE_PROMPT"],
              },
              inpaint: {
                desc: "Inpaint an image with a mask and prompt",
                keys: ["SD_INPAINT_URL", "SD_NEGATIVE_PROMPT"],
              },
              outpaint: {
                desc: "Extend an image in specified directions",
                keys: ["SD_OUTPAINT_HYBRID_URL", "SD_INPAINT_URL", "OUTPAINT_USE_CANNY", "OUTPAINT_USE_DEPTH", "OUTPAINT_CONTROL_WEIGHT", "OUTPAINT_HARMONIZE_STRENGTH", "OUTPAINT_USE_HIST_MATCH", "OUTPAINT_LIGHTNESS_FIX", "OUTPAINT_COLOR_MATCH"],
              },
              soupysearch: {
                desc: "Web search via DuckDuckGo + LLM summary",
                keys: ["BEHAVIOUR_SEARCH", "SEARCH_SELECT_TEMPERATURE", "SEARCH_SUMMARY_TEMPERATURE", "LOCAL_CHAT"],
              },
              soupyimage: {
                desc: "DuckDuckGo image search",
                keys: [],
              },
              soupypost: {
                desc: "Daily article posting to Discord channels",
                keys: ["DAILY_POST_ENABLED", "DAILY_POST_CHANNELS", "DAILY_POST_ACTIVE_START", "DAILY_POST_ACTIVE_END", "DAILY_POST_INTERVAL_HOURS", "BEHAVIOUR_DAILY_POST"],
              },
              soupysky: {
                desc: "Bluesky engagement — replies, reposts, original posts, likes, follows",
                keys: ["BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD", "BLUESKY_AUTO_REPLY", "BLUESKY_REPLIES_MIN", "BLUESKY_REPLIES_MAX", "BLUESKY_REPOSTS_PER_DAY", "BLUESKY_POSTS_PER_DAY"],
              },
              soupyself: {
                desc: "Self-knowledge and reflection system",
                keys: ["SELF_MD_ENABLED", "SELF_MD_REFLECT_INTERVAL_HOURS", "SELF_MD_MIN_INTERACTIONS", "SELF_MD_MAX_WORDS", "SELF_MD_CORE_MAX_WORDS", "SELF_MD_ARCHIVE_MAX_CHARS", "SELF_MD_MAX_ACCUMULATED", "SELF_MD_REFLECT_TEMPERATURE", "SELF_MD_REFLECT_MAX_TOKENS", "SELF_MD_CORE_TEMPERATURE", "SELF_MD_CORE_MAX_TOKENS"],
              },
              soupyscan: {
                desc: "Archive channel messages to database",
                keys: ["SOUPY_DB_DIR", "SCAN_EXCLUDE_CHANNEL_IDS", "FIRST_SCAN_LOOKBACK_DAYS", "ARCHIVE_AUTO_SCAN_POLL_SECONDS", "RAG_REINDEX_INTERVAL_HOURS"],
              },
              soupystats: {
                desc: "Server and bot statistics",
                keys: [],
              },
              soupymuse: {
                desc: "Soupy thinks out loud in a channel",
                keys: ["MUSING_ENABLED", "MUSING_CHANNEL_ID", "MUSING_POLL_MINUTES_MIN", "MUSING_POLL_MINUTES_MAX", "MUSING_CHANCE"],
              },
              "8ball": {
                desc: "Classic Magic 8-Ball responses",
                keys: [],
              },
              "9ball": {
                desc: "LLM-powered mystical 9-ball",
                keys: ["9BALL", "NINE_BALL_TEMPERATURE", "LOCAL_CHAT"],
              },
              status: {
                desc: "Bot and service health check",
                keys: [],
              },
              helpsoupy: {
                desc: "Display all available commands",
                keys: [],
              },
              whattime: {
                desc: "Current time in any city",
                keys: ["TIMEZONE"],
              },
              weather: {
                desc: "Current weather for a location",
                keys: [],
              },
              testurl: {
                desc: "Test URL content extraction",
                keys: ["URL_FETCH_TIMEOUT", "URL_MAX_CONTENT_LENGTH"],
              },
            };

            var CMD_DEFAULTS = {
              SD_SERVER_URL: "http://localhost:8000/",
              SD_IMG2IMG_URL: "http://localhost:8001/sd_img2img",
              SD_INPAINT_URL: "http://localhost:8001/sd_inpaint",
              SD_OUTPAINT_HYBRID_URL: "(SD_SERVER_URL)/outpaint_hybrid",
              SD_STEPS: "30",
              SD_GUIDANCE: "7.5",
              SD_NEGATIVE_PROMPT: "(built-in negative prompt)",
              SD_DEFAULT_WIDTH: "1024",
              SD_DEFAULT_HEIGHT: "1024",
              SD_WIDE_WIDTH: "1280",
              SD_WIDE_HEIGHT: "768",
              SD_TALL_WIDTH: "768",
              SD_TALL_HEIGHT: "1280",
              LOCAL_CHAT: "(required)",
              SEARCH_SELECT_TEMPERATURE: "0.3",
              SEARCH_SUMMARY_TEMPERATURE: "0.7",
              BEHAVIOUR_SEARCH: "(built-in search persona)",
              NINE_BALL_TEMPERATURE: "0.9",
              "9BALL": "(built-in 9-ball prompt)",
              DAILY_POST_ENABLED: "false",
              DAILY_POST_CHANNELS: '{"channel_id": "topic hint"}',
              DAILY_POST_ACTIVE_START: "8",
              DAILY_POST_ACTIVE_END: "18",
              DAILY_POST_INTERVAL_HOURS: "24",
              BEHAVIOUR_DAILY_POST: "(built-in daily post persona)",
              MUSING_ENABLED: "false",
              MUSING_CHANNEL_ID: "(channel ID)",
              MUSING_POLL_MINUTES_MIN: "10",
              MUSING_POLL_MINUTES_MAX: "20",
              MUSING_CHANCE: "0.10",
              SELF_MD_ENABLED: "false",
              SELF_MD_REFLECT_INTERVAL_HOURS: "24",
              SELF_MD_MIN_INTERACTIONS: "3",
              SELF_MD_MAX_WORDS: "15000",
              SELF_MD_CORE_MAX_WORDS: "800",
              SELF_MD_ARCHIVE_MAX_CHARS: "50000",
              SELF_MD_MAX_ACCUMULATED: "60",
              SELF_MD_REFLECT_TEMPERATURE: "0.7",
              SELF_MD_REFLECT_MAX_TOKENS: "4000",
              SELF_MD_CORE_TEMPERATURE: "0.5",
              SELF_MD_CORE_MAX_TOKENS: "1500",
              SOUPY_DB_DIR: "soupy_database/databases",
              SCAN_EXCLUDE_CHANNEL_IDS: "(none)",
              FIRST_SCAN_LOOKBACK_DAYS: "365",
              ARCHIVE_AUTO_SCAN_POLL_SECONDS: "45",
              RAG_REINDEX_INTERVAL_HOURS: "6",
              TIMEZONE: "UTC",
              URL_FETCH_TIMEOUT: "10",
              URL_MAX_CONTENT_LENGTH: "50000",
              OUTPAINT_USE_CANNY: "true",
              OUTPAINT_USE_DEPTH: "true",
              OUTPAINT_CONTROL_WEIGHT: "0.4",
              OUTPAINT_HARMONIZE_STRENGTH: "0.35",
              OUTPAINT_USE_HIST_MATCH: "true",
              OUTPAINT_LIGHTNESS_FIX: "true",
              OUTPAINT_COLOR_MATCH: "true",
            };

            var ALL_COMMANDS = Object.keys(COMMAND_CONFIG);
            var disabled = runtimeFlags.disabled_commands || [];
            var MULTILINE = new Set(["BEHAVIOUR_SEARCH", "BEHAVIOUR_DAILY_POST", "9BALL", "SD_NEGATIVE_PROMPT", "DAILY_POST_CHANNELS"]);

            function toggleCmd(cmd) {
              var isDisabled = disabled.indexOf(cmd) !== -1;
              var newList = isDisabled
                ? disabled.filter(function (c) { return c !== cmd; })
                : disabled.concat([cmd]);
              setBusy(true);
              setNoticeError(false);
              readJson("/api/runtime-flags", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ disabled_commands: newList }),
              })
                .then(function (data) {
                  setRuntimeFlags(data || {});
                  setNotice("/" + cmd + (isDisabled ? " enabled." : " disabled."));
                })
                .catch(function (err) {
                  setNotice("Failed: " + err.message);
                  setNoticeError(true);
                })
                .finally(function () { setBusy(false); });
            }

            function openCmdConfig(cmd) {
              var cfg = COMMAND_CONFIG[cmd];
              if (!cfg || !cfg.keys.length) return;
              var drafts = {};
              cfg.keys.forEach(function (k) { drafts[k] = envVars[k] || ""; });
              setCmdEnvDrafts(drafts);
              setSelectedCommand(selectedCommand === cmd ? null : cmd);

              // For soupypost: load channel list and parse existing config
              if (cmd === "soupypost" && dbGuild) {
                readJson("/api/channels/" + dbGuild).then(function (data) {
                  setCmdChannelList(data.channels || []);
                }).catch(function () { setCmdChannelList([]); });
                // Parse existing DAILY_POST_CHANNELS JSON
                try {
                  var parsed = JSON.parse(envVars.DAILY_POST_CHANNELS || "{}");
                  setCmdChannelConfig(typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {});
                } catch (_) { setCmdChannelConfig({}); }
              }
            }

            function saveCmdConfig(cmd) {
              var payload = {};
              var cfg = COMMAND_CONFIG[cmd];
              cfg.keys.forEach(function (k) {
                // For soupypost, DAILY_POST_CHANNELS is built from the channel picker
                if (cmd === "soupypost" && k === "DAILY_POST_CHANNELS") {
                  var chJson = JSON.stringify(cmdChannelConfig);
                  if (chJson !== "{}") payload[k] = chJson;
                  return;
                }
                var val = (cmdEnvDrafts[k] || "").trim();
                if (val) payload[k] = val;
              });
              setBusy(true);
              setNoticeError(false);
              readJson("/api/env/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              })
                .then(function () {
                  // Reload env to pick up changes
                  return readJson("/api/env/get");
                })
                .then(function (data) {
                  setEnvVars(data.vars || {});
                  setNotice("/" + cmd + " settings saved.");
                })
                .catch(function (err) {
                  setNotice("Save failed: " + err.message);
                  setNoticeError(true);
                })
                .finally(function () { setBusy(false); });
            }

            return e(
              "article",
              null,
              e("h4", { style: { marginTop: 0 } }, "Slash Commands"),
              e(
                "p",
                { className: "muted", style: { marginTop: 0 } },
                "Toggle commands on/off. Click a command name to view and edit its settings."
              ),
              e(
                "div",
                { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "0.35rem 0.75rem" } },
                ALL_COMMANDS.map(function (cmd) {
                  var isEnabled = disabled.indexOf(cmd) === -1;
                  var cfg = COMMAND_CONFIG[cmd];
                  var hasSettings = cfg && cfg.keys.length > 0;
                  var isSelected = selectedCommand === cmd;
                  return e(
                    "div",
                    { key: cmd, style: { display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.92rem" } },
                    e("input", {
                      type: "checkbox",
                      checked: isEnabled,
                      disabled: busy,
                      onChange: function () { toggleCmd(cmd); },
                      style: { cursor: "pointer" },
                    }),
                    hasSettings
                      ? e("a", {
                          href: "#",
                          className: "mono",
                          onClick: function (ev) { ev.preventDefault(); openCmdConfig(cmd); },
                          style: {
                            color: isSelected ? "var(--tab-active-text, #fff)" : "var(--text-secondary, #aaa)",
                            textDecoration: isSelected ? "underline" : "none",
                            cursor: "pointer",
                            fontWeight: isSelected ? "bold" : "normal",
                          },
                        }, "/" + cmd)
                      : e("span", { className: "mono", style: { color: "var(--text-muted, #666)" } }, "/" + cmd)
                  );
                })
              ),
              // ── Command config panel ──────────────────────────
              selectedCommand && COMMAND_CONFIG[selectedCommand] && COMMAND_CONFIG[selectedCommand].keys.length > 0
                ? e(
                    "div",
                    {
                      style: {
                        marginTop: "1rem",
                        padding: "1rem",
                        background: "var(--card-bg, #1a1a2e)",
                        border: "1px solid var(--card-border, #333)",
                        borderRadius: "6px",
                      },
                    },
                    e(
                      "div",
                      { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" } },
                      e("h4", { style: { margin: 0 } }, "/" + selectedCommand + " settings"),
                      e(
                        "button",
                        {
                          className: "secondary",
                          style: { padding: "0.2rem 0.6rem", fontSize: "0.85rem" },
                          onClick: function () { setSelectedCommand(null); },
                        },
                        "Close"
                      )
                    ),
                    e("p", { className: "muted", style: { marginTop: 0, marginBottom: "0.75rem" } },
                      COMMAND_CONFIG[selectedCommand].desc
                    ),
                    COMMAND_CONFIG[selectedCommand].keys.map(function (k) {
                      // --- Channel picker for DAILY_POST_CHANNELS ---
                      if (k === "DAILY_POST_CHANNELS" && selectedCommand === "soupypost") {
                        return e(
                          "div",
                          { key: k, className: "dash-field", style: { marginBottom: "0.75rem" } },
                          e("label", { className: "mono", style: { fontSize: "0.85rem" } }, "Channels & Topics"),
                          !cmdChannelList.length
                            ? e("p", { className: "muted", style: { margin: "0.3rem 0" } },
                                dbGuild
                                  ? "No channels found. Run a scan first."
                                  : "Select a guild in the RAG section above to load channels."
                              )
                            : e(
                                "div",
                                { style: { display: "flex", flexDirection: "column", gap: "0.35rem", marginTop: "0.3rem" } },
                                cmdChannelList.map(function (ch) {
                                  var isChecked = ch.id in cmdChannelConfig;
                                  var topicVal = cmdChannelConfig[ch.id] || "";
                                  return e(
                                    "div",
                                    {
                                      key: ch.id,
                                      style: {
                                        display: "flex", alignItems: "center", gap: "0.5rem",
                                        padding: "0.3rem 0.5rem", borderRadius: "4px",
                                        background: isChecked ? "var(--card-bg, #1a1a2e)" : "transparent",
                                        border: isChecked ? "1px solid var(--card-border, #333)" : "1px solid transparent",
                                      },
                                    },
                                    e("input", {
                                      type: "checkbox",
                                      checked: isChecked,
                                      onChange: function () {
                                        setCmdChannelConfig(function (prev) {
                                          var n = Object.assign({}, prev);
                                          if (isChecked) { delete n[ch.id]; }
                                          else { n[ch.id] = ch.name; }
                                          return n;
                                        });
                                      },
                                      style: { cursor: "pointer" },
                                    }),
                                    e("span", { className: "mono", style: { minWidth: "140px", fontSize: "0.88rem" } },
                                      "#" + ch.name
                                    ),
                                    e("span", { className: "muted", style: { fontSize: "0.8rem", minWidth: "60px" } },
                                      fmtNum(ch.messages) + " msgs"
                                    ),
                                    isChecked
                                      ? e("input", {
                                          type: "text",
                                          value: topicVal,
                                          placeholder: "topic hint (e.g. 'tech news', 'politics')",
                                          onChange: function (ev) {
                                            var val = ev.target.value;
                                            setCmdChannelConfig(function (prev) {
                                              var n = Object.assign({}, prev);
                                              n[ch.id] = val;
                                              return n;
                                            });
                                          },
                                          style: { flex: 1, fontFamily: "monospace", fontSize: "0.85rem" },
                                        })
                                      : null
                                  );
                                })
                              )
                        );
                      }
                      // --- Normal fields ---
                      var isMultiline = MULTILINE.has(k);
                      var placeholder = CMD_DEFAULTS[k] ? "default: " + CMD_DEFAULTS[k] : "";
                      var fieldStyle = { width: "100%", fontFamily: "monospace", fontSize: "0.85rem" };
                      var onChange = function (ev) {
                        setCmdEnvDrafts(function (prev) {
                          var n = Object.assign({}, prev);
                          n[k] = ev.target.value;
                          return n;
                        });
                      };
                      return e(
                        "div",
                        { key: k, className: "dash-field", style: { marginBottom: "0.5rem" } },
                        e("label", { className: "mono", style: { fontSize: "0.85rem" } }, k),
                        isMultiline
                          ? e("textarea", {
                              value: cmdEnvDrafts[k] || "",
                              placeholder: placeholder,
                              onChange: onChange,
                              style: Object.assign({}, fieldStyle, { minHeight: "80px" }),
                            })
                          : e("input", {
                              type: "text",
                              value: cmdEnvDrafts[k] || "",
                              placeholder: placeholder,
                              onChange: onChange,
                              style: fieldStyle,
                            })
                      );
                    }),
                    e(
                      "div",
                      { className: "dash-controls", style: { marginTop: "0.75rem" } },
                      e(
                        "button",
                        {
                          disabled: busy,
                          onClick: function () { saveCmdConfig(selectedCommand); },
                        },
                        busy ? "Saving..." : "Save settings"
                      ),
                      e(
                        "button",
                        {
                          className: "secondary",
                          disabled: busy,
                          onClick: function () { openCmdConfig(selectedCommand); },
                        },
                        "Revert"
                      )
                    )
                  )
                : null
            );
          })();

        var bandKnobs = e(
          "div",
          { className: "dash-knobs" },
          e("div", { className: "dash-knob" },
            e("div", {
              className: "dash-knob-header",
              role: "button",
              tabIndex: 0,
              "aria-expanded": overviewKnobs.commands ? "true" : "false",
              onClick: knobToggle("commands"),
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); knobToggle("commands")(); } }
            },
              e("span", null, "Slash Commands"),
              e("span", { className: "dash-knob-chevron" }, "▶")
            ),
            overviewKnobs.commands
              ? e("div", { className: "dash-knob-body" }, slashCommandsBody)
              : null
          ),
          e("div", { className: "dash-knob" },
            e("div", {
              className: "dash-knob-header",
              role: "button",
              tabIndex: 0,
              "aria-expanded": overviewKnobs.rag ? "true" : "false",
              onClick: knobToggle("rag"),
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); knobToggle("rag")(); } }
            },
              e("span", null, "RAG (archived messages)"),
              e("span", { className: "dash-knob-chevron" }, "▶")
            ),
            overviewKnobs.rag
              ? e("div", { className: "dash-knob-body" }, ragKnobBody)
              : null
          ),
          e("div", { className: "dash-knob" },
            e("div", {
              className: "dash-knob-link",
              role: "button",
              tabIndex: 0,
              onClick: function () { setTab("model"); },
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setTab("model"); } }
            },
              e("span", null, "Model & Personality"),
              e("span", { className: "dash-knob-link-arrow" }, "→")
            )
          ),
          e("div", { className: "dash-knob" },
            e("div", {
              className: "dash-knob-link",
              role: "button",
              tabIndex: 0,
              onClick: function () { setTab("database"); },
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setTab("database"); } }
            },
              e("span", null, "Database & Archive"),
              e("span", { className: "dash-knob-link-arrow" }, "→")
            )
          )
        );

        return e(
          React.Fragment,
          null,
          stickyBar,
          bandLive,
          bandLoops,
          bandSnapshot,
          bandKnobs
        );
      })(),
      tab === "model" &&
        e(
          "section",
          { className: "dash-app" },
          e(
            "div",
            { className: "dash-grid-2-equal" },
            e(
              "article",
              null,
              e("h4", { style: { marginTop: 0 } }, "LLM Model Selection"),
              e(
                "div",
                { className: "dash-field" },
                e("label", null, "Find model"),
                e("input", {
                  placeholder: "Type to filter models...",
                  value: modelQuery,
                  onChange: function (ev) {
                    setModelQuery(ev.target.value);
                  },
                }),
                e("label", null, "Chat model"),
                e(
                  "select",
                  {
                    value: selectedModel,
                    onChange: function (ev) {
                      setSelectedModel(ev.target.value);
                    },
                  },
                  filteredModels.map(function (m) {
                    var label = (window._lmModelLabels && window._lmModelLabels[m]) || m;
                    return e("option", { key: m, value: m }, label);
                  })
                ),
                e("label", { style: { marginTop: "0.5rem" } }, "Context window (tokens)"),
                e("input", {
                  type: "number",
                  min: "1024",
                  max: "131072",
                  step: "1024",
                  value: contextWindowSize,
                  onChange: function (ev) {
                    setContextWindowSize(ev.target.value);
                  },
                  style: { width: "8rem" },
                }),
              ),
              e("p", { className: "muted" }, "Switching models will unload the current model in LM Studio and load the new one with the specified context window."),
              e(
                "button",
                {
                  disabled: busy || !selectedModel,
                  onClick: async function () {
                    setBusy(true);
                    setNotice("");
                    setNoticeError(false);
                    try {
                      // Step 1: Tell LM Studio to unload old model and load new one
                      // Step 1: Try to tell LM Studio to switch models
                      var lmSwitchOk = false;
                      var lmMessage = "";
                      try {
                        var resp = await fetch("/api/lm-studio/switch-model", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            model: selectedModel,
                            context_length: parseInt(contextWindowSize, 10) || 16000,
                          }),
                        });
                        var result = await resp.json();
                        lmSwitchOk = result.ok;
                        lmMessage = result.message || "";
                      } catch (lmErr) {
                        lmMessage = lmErr.message;
                      }

                      // Step 2: ALWAYS update .env-stable and restart bot, even if LM Studio call failed
                      await fetch("/api/env/save", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          LOCAL_CHAT: selectedModel,
                          CONTEXT_WINDOW_TOKENS: contextWindowSize,
                        }),
                      });
                      await fetch("/api/bot/restart", { method: "POST" });

                      if (lmSwitchOk) {
                        setNotice("Loaded " + selectedModel + " (" + contextWindowSize + " ctx) and restarted bot.");
                      } else {
                        setNotice("Bot restarted with " + selectedModel + ". LM Studio auto-load note: " + lmMessage);
                        setNoticeError(true);
                      }
                      await loadEnv();
                    } catch (err) {
                      setNotice("Model switch failed: " + err.message);
                      setNoticeError(true);
                    } finally {
                      setBusy(false);
                    }
                  },
                },
                busy ? "Switching model..." : "Switch model + restart"
              )
            ),
            e(
              "article",
              null,
              e("h4", { style: { marginTop: 0 } }, "Model Catalog"),
              e(
                "div",
                { className: "dash-field" },
                e("label", null, "AVAILABLE_MODELS (one per line)"),
                e("textarea", {
                  value: modelCatalogDraft,
                  onChange: function (ev) {
                    setModelCatalogDraft(ev.target.value);
                  },
                  placeholder: "llama3.1:8b\nmistral:latest\nqwen2.5:14b",
                })
              ),
              e(
                "div",
                { className: "dash-controls" },
                e(
                  "button",
                  { disabled: busy, onClick: fetchLMStudioModels },
                  busy ? "Fetching..." : "Fetch from LM Studio"
                ),
                e(
                  "button",
                  { className: "secondary", disabled: busy, onClick: function () { setModelCatalogDraft(models.join("\n")); } },
                  "Reset to current list"
                ),
                e(
                  "button",
                  { disabled: busy, onClick: saveAvailableModels },
                  busy ? "Saving..." : "Save catalog + restart"
                )
              )
            )
          ),
          e(
            "article",
            { className: "personality-panel" },
            e("h4", { style: { marginTop: 0 } }, "Personality"),
            e(
              "p",
              { className: "muted", style: { marginTop: 0 } },
              "Edit the ",
              e("span", { className: "mono" }, "BEHAVIOUR"),
              " system prompt or the ",
              e("span", { className: "mono" }, "BEHAVIOUR_SEARCH"),
              " search prompt below. Auto-format on load; ",
              e("span", { className: "mono" }, "Load raw"),
              " shows the file as stored."
            ),
            e(
              "div",
              { className: "personality-toolbar dash-controls" },
              e(
                "div",
                { className: "dash-field", style: { marginBottom: 0, flex: "1 1 220px", minWidth: "min(100%, 280px)" } },
                e("label", { htmlFor: "personality-select" }, "Prompt to edit"),
                e(
                  "select",
                  {
                    id: "personality-select",
                    value: preset,
                    disabled: !personalityOptions.length,
                    onChange: function (ev) {
                      var val = ev.target.value;
                      setPreset(val);
                      setPersonalityText(getPresetValueFormatted(val));
                    },
                  },
                  personalityOptions.length
                    ? personalityOptions.map(function (opt) {
                        return e(
                          "option",
                          { key: opt.key, value: opt.key },
                          opt.label + " — " + opt.key
                        );
                      })
                    : e("option", { value: "" }, "No named presets — add one")
                )
              ),
              e(
                "button",
                {
                  type: "button",
                  className: "secondary",
                  disabled: busy,
                  onClick: function () {
                    loadEnv();
                    setNotice("Reloaded from .env");
                    setNoticeError(false);
                  },
                },
                "Reload from .env"
              )
            ),
            e("p", { className: "muted mono", style: { margin: "0.35rem 0 0.5rem" } }, "Saving to env key: " + preset),
            e(
              "div",
              { className: "dash-field" },
              e("label", { htmlFor: "personality-body" }, "Behaviour text"),
              e("textarea", {
                id: "personality-body",
                className: "personality-editor",
                disabled: !personalityOptions.length,
                value: personalityText,
                onChange: function (ev) {
                  setPersonalityText(normalizeLineEndings(ev.target.value));
                },
                placeholder:
                  "Sections, list numbers, and long ALL-CAPS headers are broken apart for you on load. Edit freely, then Save.",
              })
            ),
            e(
              "div",
              { className: "muted" },
              fmtNum(personalityText.split("\n").length),
              " lines · ",
              fmtNum(personalityText.length),
              " characters"
            ),
            e(
              "div",
              { className: "dash-controls", style: { marginTop: "0.65rem" } },
              e(
                "button",
                {
                  type: "button",
                  className: "secondary",
                  disabled: busy,
                  onClick: function () {
                    setPersonalityText(getPresetValueFormatted(preset));
                  },
                },
                "Revert to saved (formatted)"
              ),
              e(
                "button",
                {
                  type: "button",
                  className: "secondary",
                  disabled: busy,
                  onClick: function () {
                    setPersonalityText(getPresetValueRaw(preset));
                  },
                },
                "Load raw from .env"
              ),
              e(
                "button",
                {
                  type: "button",
                  className: "secondary",
                  disabled: busy,
                  onClick: function () {
                    setPersonalityText(formatBehaviourForEditor(personalityText));
                  },
                },
                "Re-tidy editor"
              ),
              e(
                "button",
                {
                  type: "button",
                  className: "secondary",
                  disabled: busy,
                  onClick: function () {
                    var payload = {};
                    payload[preset] = normalizeLineEndings(personalityText);
                    saveEnvOnly(payload, preset + " saved to .env (bot not restarted).");
                  },
                },
                busy ? "Saving..." : "Save to .env"
              ),
              e(
                "button",
                {
                  type: "button",
                  disabled: busy,
                  onClick: function () {
                    var payload = {};
                    payload[preset] = normalizeLineEndings(personalityText);
                    saveEnvAndRestart(payload, titleFromPersonalityKey(preset) + " saved; bot restarted.");
                  },
                },
                busy ? "Saving..." : "Save preset + restart bot"
              ),
            )
          )
        ),
      tab === "stats" &&
        e(StatsStudio, {
          summary: summary,
          active: tab === "stats",
          onReload: loadSummary,
        }),
      tab === "archive" && e(MediaArchiveTab, { active: tab === "archive" }),
      tab === "database" &&
        e(
          "div",
          { className: "database-tab-stack" },
          e(
            "article",
            null,
            e("h4", { style: { marginTop: 0 } }, "Profile batch"),
            e(
              "p",
              { className: "muted", style: { maxWidth: "52rem", marginTop: "0.35rem" } },
              "Builds structured profiles (topics, tone, channels, ",
              e("strong", null, "relationships with other members"),
              ") from the archive. Pause stops before the next user; Resume continues. The activity log below shows each step (archive read → LLM call → save)."
            ),
            e(
              "p",
              { className: "muted", style: { fontSize: "0.88rem" } },
              "Guild: ",
              dbGuild ? e("span", { className: "mono" }, dbGuild) : e("span", null, "choose in Database Explorer below.")
            ),
            profBatchStatus && profBatchStatus.ok
              ? e(
                  "div",
                  { className: "profile-batch-status-bar" },
                  e(
                    "span",
                    {
                      className:
                        "profile-batch-status-chip" +
                        (profBatchStatus.task_running || profBatchStatus.status === "running"
                          ? " profile-batch-status-chip-live"
                          : profBatchStatus.status === "paused"
                            ? " profile-batch-status-chip-paused"
                            : ""),
                    },
                    String(profBatchStatus.status || "—")
                  ),
                  e(
                    "span",
                    { className: "profile-batch-status-meta" },
                    (function () {
                      var t = profBatchStatus.total || 0;
                      var ni = profBatchStatus.next_index != null ? Number(profBatchStatus.next_index) : 0;
                      var bs = profBatchStatus.batch_stats || {};
                      var saved = bs.saved != null ? Number(bs.saved) : null;
                      var skipped = bs.skipped != null ? Number(bs.skipped) : null;
                      var failed = bs.failed != null ? Number(bs.failed) : null;
                      var prc =
                        typeof profBatchStatus.profile_row_count === "number"
                          ? profBatchStatus.profile_row_count
                          : null;
                      var parts = [];
                      parts.push("queue " + fmtNum(ni) + " / " + fmtNum(t) + " (candidates visited)");
                      if (prc !== null && !Number.isNaN(prc)) {
                        parts.push(fmtNum(prc) + " row(s) in user_profile_summaries");
                      }
                      if (
                        profBatchStatus.batch_stats &&
                        saved !== null &&
                        skipped !== null &&
                        failed !== null &&
                        !Number.isNaN(saved + skipped + failed)
                      ) {
                        parts.push(
                          "this run: " +
                            fmtNum(saved) +
                            " saved · " +
                            fmtNum(skipped) +
                            " skipped · " +
                            fmtNum(failed) +
                            " failed"
                        );
                      }
                      return parts.join(" · ") + (profBatchStatus.task_running ? " · worker running" : "");
                    })()
                  )
                )
              : null,
            e(
              "p",
              { className: "muted", style: { fontSize: "0.8rem", marginTop: "0.35rem" } },
              "Log refresh: about every ",
              e("strong", null, "12s"),
              " while a job is active, ",
              e("strong", null, "2 min"),
              " when idle — so the server log stays quiet."
            ),
            e(
              "div",
              { className: "dash-controls", style: { flexWrap: "wrap", gap: "0.5rem", marginTop: "0.5rem" } },
              e(
                "button",
                {
                  className: "secondary",
                  disabled: !dbGuild || busy,
                  onClick: profileResetAll,
                },
                "Delete all profiles"
              ),
              e(
                "button",
                { disabled: !dbGuild || busy, onClick: profileBatchStart },
                "Start batch"
              ),
              e(
                "button",
                { className: "secondary", disabled: !dbGuild, onClick: profileBatchPause },
                "Pause"
              ),
              e(
                "button",
                { className: "secondary", disabled: !dbGuild, onClick: profileBatchResume },
                "Resume"
              ),
              e(
                "button",
                { className: "secondary", disabled: !dbGuild, onClick: profileBatchCancel },
                "Cancel worker"
              )
            ),
            e(
              "p",
              { className: "muted", style: { fontSize: "0.82rem", marginTop: "0.35rem" } },
              e("span", { className: "mono" }, "USER_PROFILES_BATCH_MAX_USERS"),
              ", ",
              e("span", { className: "mono" }, "USER_PROFILE_SAMPLE_MESSAGES"),
              ", ",
              e("span", { className: "mono" }, "USER_PROFILE_MAX_TOKENS"),
              " — see ",
              e("span", { className: "mono" }, ".env-stable"),
              "."
            ),
            e(
              "div",
              { className: "profile-batch-console" },
              e(
                "div",
                { className: "profile-batch-console-head" },
                e("span", { className: "profile-batch-console-title" }, "Activity log"),
                e(
                  "span",
                  { className: "profile-batch-console-sub" },
                  "newest at bottom · auto-scroll"
                )
              ),
              e(
                "div",
                {
                  ref: profBatchLogRef,
                  className: "profile-batch-console-body mono",
                  role: "log",
                  "aria-live": "polite",
                  "aria-relevant": "additions",
                },
                (profBatchLogLines || []).length
                  ? (profBatchLogLines || []).map(function (ln, i) {
                      return profileBatchLogLineRow(ln, i);
                    })
                  : e(
                      "div",
                      { className: "profile-batch-console-empty" },
                      "Start a batch to see step-by-step progress: sampling messages, calling LOCAL_CHAT, saving the profile row."
                    )
              )
            )
          ),
          e(
            "article",
            null,
            e("h4", { style: { marginTop: 0 } }, "Database Explorer"),
          e(
            "div",
            { className: "db-toolbar" },
            e("div", { className: "dash-field" }, e("label", null, "Server"), e(
              "select",
              {
                value: dbGuild,
                onChange: function (ev) { setDbGuild(ev.target.value); },
              },
              e("option", { value: "" }, "Choose a server"),
              (dbStatus.databases || []).map(function (d) {
                const label = (d.guild_name || "Guild") + " (" + d.guild_id + ")";
                return e("option", { key: d.guild_id, value: d.guild_id }, label);
              })
            )),
            e("div", { className: "dash-field" }, e("label", null, "Table"), e(
              "select",
              { value: dbTable, onChange: function (ev) { setDbTable(ev.target.value); } },
              e("option", { value: "messages" }, "messages"),
              e("option", { value: "scan_metadata" }, "scan_metadata")
            )),
            e("div", { className: "dash-field" }, e("label", null, "Search"), e("input", { value: dbSearch, placeholder: "content, channel, user", onChange: function (ev) { setDbSearch(ev.target.value); } })),
            e("div", { className: "dash-field" }, e("label", null, "Channel ID"), e("input", { value: dbChannel, placeholder: "optional", onChange: function (ev) { setDbChannel(ev.target.value); } })),
            e("div", { className: "dash-field" }, e("label", null, "Pick user"), e(
              "select",
              {
                value: pickerSelectValue(dbPickerUsers, dbUser),
                onChange: function (ev) {
                  setDbUser(ev.target.value);
                },
                disabled: !dbGuild || dbTable !== "messages",
              },
              e(
                "option",
                { value: "" },
                !dbGuild
                  ? "Choose a server first"
                  : dbTable !== "messages"
                    ? "Only for messages table"
                    : "— anyone —"
              ),
              dbPickerUsers.map(function (u) {
                return e(
                  "option",
                  { key: "arch-" + u.user_id, value: String(u.user_id) },
                  u.label + " · " + String(u.user_id) + " · " + fmtNum(u.message_count) + " msgs"
                );
              })
            )),
            e("div", { className: "dash-field" }, e("label", null, "User ID"), e("input", { value: dbUser, placeholder: "optional, or pick above", onChange: function (ev) { setDbUser(ev.target.value); } })),
            e("div", { className: "dash-field" }, e("label", null, "Rows"), e(
              "select",
              {
                value: String(dbLimit),
                onChange: function (ev) { setDbLimit(Number(ev.target.value || 50)); },
              },
              [25, 50, 100, 200].map(function (n) {
                return e("option", { key: n, value: String(n) }, String(n));
              })
            ))
          ),
          e(
            "div",
            { className: "db-switches", style: { marginBottom: "0.5rem" } },
            e("label", null, e("input", { type: "checkbox", checked: dbHasImages, onChange: function () { setDbHasImages(!dbHasImages); } }), " with image descriptions"),
            e("label", null, e("input", { type: "checkbox", checked: dbHasUrls, onChange: function () { setDbHasUrls(!dbHasUrls); } }), " with URL summaries")
          ),
          e(
            "div",
            { className: "dash-controls", style: { flexWrap: "wrap", gap: "0.5rem", alignItems: "center" } },
            e("button", { onClick: function () { loadRows(0); } }, "Load rows"),
            e("button", { className: "secondary", onClick: function () { loadRows(Math.max(0, dbOffset - dbLimit)); } }, "Prev"),
            e("button", { className: "secondary", onClick: function () { if (dbOffset + dbLimit < dbTotal) loadRows(dbOffset + dbLimit); } }, "Next"),
            e("button", { className: "secondary", onClick: triggerScan }, "Run incremental scan")
          ),
          e(
            "div",
            {
              className: "dash-controls",
              style: { marginTop: "0.65rem", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" },
            },
            e(
              "label",
              { style: { display: "flex", alignItems: "center", gap: "0.4rem" } },
              "Auto archive every",
              e("input", {
                type: "number",
                min: 0,
                max: 10080,
                step: 1,
                style: { width: "5rem" },
                value: archiveScanMinutes,
                onChange: function (ev) {
                  setArchiveScanMinutes(ev.target.value);
                },
              }),
              e("span", { className: "muted" }, "minutes (0 = off). Example: 60 ≈ hourly. RAG rebuilds after each scan.")
            ),
            e("button", { className: "secondary", disabled: !dbGuild, onClick: saveArchiveSchedule }, "Save schedule")
          ),
          e("p", { className: "muted", style: { marginTop: "0.5rem" } }, dbInfo),
          e(
            "div",
            { className: "db-table-wrap" },
            e(
              "table",
              { className: "db-table" },
              e(
                "thead",
                null,
                dbTable === "scan_metadata"
                  ? e("tr", null, e("th", null, "Type"), e("th", null, "Messages"), e("th", null, "Last scan"), e("th", null, "Created"))
                  : e("tr", null, e("th", null, "Timestamp"), e("th", null, "User"), e("th", null, "Channel"), e("th", null, "Content"), e("th", null, "Signals"))
              ),
              e(
                "tbody",
                null,
                dbRows.map(function (r, i) {
                  if (dbTable === "scan_metadata") {
                    return e(
                      "tr",
                      { key: i, onClick: function () { setDbSelectedRow(r); } },
                      e("td", null, r.scan_type || ""),
                      e("td", null, fmtNum(r.messages_scanned)),
                      e("td", { className: "mono" }, r.last_scan_time || ""),
                      e("td", { className: "mono" }, r.created_at || "")
                    );
                  }
                  return e(
                    "tr",
                    { key: i, onClick: function () { setDbSelectedRow(r); } },
                    e("td", { className: "mono" }, (r.date || "") + " " + (r.time || "")),
                    e("td", null, e("div", null, r.username || "Unknown"), e("div", { className: "muted mono" }, r.user_id || "")),
                    e("td", null, e("div", null, r.channel_name || "Unknown"), e("div", { className: "muted mono" }, r.channel_id || "")),
                    e("td", null, r.message_content || e("span", { className: "muted" }, "(empty)")),
                    e(
                      "td",
                      null,
                      r.image_description ? e("span", { className: "pill" }, "image") : null,
                      r.url_summary ? e("span", { className: "pill" }, "url") : null,
                      !r.image_description && !r.url_summary ? e("span", { className: "muted" }, "none") : null
                    )
                  );
                })
              )
            )
          ),
          dbSelectedRow &&
            e(
              "div",
              { className: "row-drawer", style: { marginTop: "0.65rem" } },
              e("h5", { style: { marginTop: 0 } }, "Row details"),
              e("pre", { className: "mono", style: { margin: 0, whiteSpace: "pre-wrap" } }, JSON.stringify(dbSelectedRow, null, 2))
            )
          ),
          e(
            "article",
            { style: { marginTop: "2rem" } },
            e("h4", { style: { marginTop: 0 } }, "Member profiles (RAG sketches)"),
            e(
              "p",
              { className: "muted", style: { marginTop: "0.35rem", maxWidth: "52rem" } },
              "Rows from ",
              e("strong", null, "user_profile_summaries"),
              ". Built from ",
              e("strong", null, "Profile batch"),
              " above. Soft context for the model—archived ",
              e("span", { className: "mono" }, "messages"),
              " remain the source of truth for what someone actually said."
            ),
            e(
              "p",
              { className: "muted", style: { marginTop: "0.5rem", fontSize: "0.88rem" } },
              "Guild: ",
              dbGuild
                ? e("span", { className: "mono" }, dbGuild)
                : e("span", null, "select one in Database Explorer above.")
            ),
            e(
              "div",
              { className: "db-toolbar", style: { marginTop: "0.5rem" } },
              e("div", { className: "dash-field" }, e("label", null, "Pick profile"), e(
                "select",
                {
                  value: pickerSelectValue(profPickerUsers, profSearch),
                  onChange: function (ev) {
                    var v = ev.target.value;
                    setProfSearch(v);
                    setProfSelected(null);
                    if (v) loadProfileList(0);
                  },
                  disabled: !dbGuild || userPickerLoading,
                },
                e("option", { value: "" }, (function () {
                  if (!dbGuild) return "Choose a server first";
                  if (userPickerLoading) return "Loading user lists…";
                  if (userPickerError) return "— list failed (see below) —";
                  if ((profPickerUsers || []).length) return "— all (use search below) —";
                  if (profileRowCount === 0) {
                    return "No rows yet — use Profile batch above";
                  }
                  return "— none loaded —";
                })()),
                (profPickerUsers || []).map(function (u) {
                  var hint = (u.label || u.nickname_hint || "").trim();
                  var label = hint ? hint + " · " + String(u.user_id) : String(u.user_id);
                  return e("option", { key: "prof-p-" + u.user_id, value: String(u.user_id) }, label);
                })
              )),
              e("div", { className: "dash-field" }, e("label", null, "Search"), e(
                "input",
                {
                  value: profSearch,
                  placeholder: "user id, nickname, or summary text",
                  onChange: function (ev) {
                    setProfSearch(ev.target.value);
                  },
                }
              )),
              e("div", { className: "dash-field" }, e("label", null, "Rows"), e(
                "select",
                {
                  value: String(profLimit),
                  onChange: function (ev) {
                    setProfLimit(Number(ev.target.value || 50));
                  },
                },
                [25, 50, 100, 200].map(function (n) {
                  return e("option", { key: "pl-" + n, value: String(n) }, String(n));
                })
              ))
            ),
            profileRowCount !== null && !userPickerError
              ? e(
                  "p",
                  { className: "muted", style: { fontSize: "0.88rem", marginTop: "0.4rem", maxWidth: "48rem" } },
                  e("span", { className: "mono" }, "user_profile_summaries"),
                  ": ",
                  e("strong", null, fmtNum(profileRowCount)),
                  profileRowCount === 0
                    ? " row — run Profile batch above (LOCAL_CHAT + archive required)."
                    : " row(s). Pick a user above or use Search."
                )
              : null,
            userPickerError
              ? e(
                  "p",
                  {
                    className: "dash-message error",
                    style: { marginTop: "0.4rem", fontSize: "0.9rem" },
                  },
                  userPickerError
                )
              : null,
            e(
              "div",
              { className: "dash-controls", style: { flexWrap: "wrap", gap: "0.5rem", alignItems: "center", marginTop: "0.5rem" } },
              e(
                "button",
                {
                  className: "secondary",
                  disabled: !dbGuild || userPickerLoading,
                  onClick: function () {
                    loadUserPickerData();
                  },
                },
                userPickerLoading ? "…" : "Refresh user lists"
              ),
              e(
                "button",
                { disabled: !dbGuild, onClick: function () { loadProfileList(0); } },
                "Load profiles"
              ),
              e(
                "button",
                {
                  className: "secondary",
                  disabled: !dbGuild || profOffset <= 0,
                  onClick: function () {
                    loadProfileList(Math.max(0, profOffset - profLimit));
                  },
                },
                "Prev"
              ),
              e(
                "button",
                {
                  className: "secondary",
                  disabled: !dbGuild || profOffset + profLimit >= profTotal,
                  onClick: function () {
                    if (profOffset + profLimit < profTotal) loadProfileList(profOffset + profLimit);
                  },
                },
                "Next"
              )
            ),
            e("p", { className: "muted", style: { marginTop: "0.5rem" } }, profInfo),
            e(
              "div",
              { className: "db-table-wrap" },
              e(
                "table",
                { className: "db-table" },
                e(
                  "thead",
                  null,
                  e(
                    "tr",
                    null,
                    e("th", null, "User ID"),
                    e("th", null, "Nick hint"),
                    e("th", null, "Msgs"),
                    e("th", null, "Updated"),
                    e("th", { style: { minWidth: "14rem" } }, "Summary (preview)")
                  )
                ),
                e(
                  "tbody",
                  null,
                  !profRows.length
                    ? e(
                        "tr",
                        null,
                        e(
                          "td",
                          { colSpan: 5, className: "muted" },
                          dbGuild ? "No rows loaded. Click Load profiles." : "Select a server in Database Explorer first, or pick one above."
                        )
                      )
                    : profRows.map(function (r, i) {
                        return e(
                          "tr",
                          {
                            key: (r.user_id || "") + "-" + i,
                            onClick: function () {
                              setProfSelected(r);
                            },
                          },
                          e("td", { className: "mono" }, r.user_id != null ? String(r.user_id) : ""),
                          e("td", null, r.nickname_hint || e("span", { className: "muted" }, "—")),
                          e("td", null, fmtNum(r.source_message_count)),
                          e("td", { className: "mono" }, r.updated_at || "—"),
                          e(
                            "td",
                            { style: { fontSize: "0.88rem", maxWidth: "28rem", whiteSpace: "normal" } },
                            previewText(r.summary, 220)
                          )
                        );
                      })
                )
              )
            ),
            profSelected &&
              e(
                "div",
                { className: "row-drawer", style: { marginTop: "0.65rem" } },
                e("h5", { style: { marginTop: 0 } }, "Profile detail"),
                e(
                  "div",
                  {
                    className: "muted mono",
                    style: { fontSize: "0.85rem", marginBottom: "0.5rem" },
                  },
                  "user_id ",
                  String(profSelected.user_id),
                  " · source_max_message_id ",
                  profSelected.source_max_message_id != null ? String(profSelected.source_max_message_id) : "—",
                  " · model ",
                  String(profSelected.model_used || "—")
                ),
                e(
                  "div",
                  {
                    style: {
                      whiteSpace: "pre-wrap",
                      fontSize: "0.92rem",
                      lineHeight: 1.45,
                      padding: "0.65rem 0.75rem",
                      borderRadius: "6px",
                      background: "rgba(0,0,0,0.06)",
                      marginBottom: "0.65rem",
                    },
                  },
                  profSelected.summary || e("span", { className: "muted" }, "(empty)")
                ),
                profSelected.structured_json
                  ? e(
                      "div",
                      { style: { marginBottom: "0.65rem" } },
                      e("h6", { style: { margin: "0.5rem 0 0.25rem" } }, "Structured (JSON)"),
                      e(
                        "pre",
                        {
                          className: "mono",
                          style: {
                            margin: 0,
                            whiteSpace: "pre-wrap",
                            fontSize: "0.78rem",
                            maxHeight: "320px",
                            overflow: "auto",
                            padding: "0.5rem",
                            background: "rgba(0,0,0,0.06)",
                          },
                        },
                        (function () {
                          try {
                            return JSON.stringify(JSON.parse(String(profSelected.structured_json || "{}")), null, 2);
                          } catch (_e) {
                            return String(profSelected.structured_json || "");
                          }
                        })()
                      )
                    )
                  : null,
                e("h6", { style: { margin: "0.5rem 0 0.25rem" } }, "Raw row"),
                e("pre", { className: "mono", style: { margin: 0, whiteSpace: "pre-wrap", fontSize: "0.8rem" } }, JSON.stringify(profSelected, null, 2))
              )
          )
        )
    );
  }

  const root = document.getElementById("dashboard-root");
  if (root) {
    ReactDOM.createRoot(root).render(e(App));
  }
})();
