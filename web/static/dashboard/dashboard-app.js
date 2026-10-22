(function () {
  // Helpers and overlay pieces shared with the tab scripts live in dashboard-core.js.
  const D = window.SoupyDash;
  const e = D.e;
  const fmtNum = D.fmtNum;
  const readJson = D.readJson;
  const { useCallback, useEffect, useMemo, useState, useRef } = React;

  const TABS = ["overview", "model", "stats", "media", "database"];
  const DB_VIEWS = ["messages", "profiles", "maintenance"];
  const GUILD_STORAGE_KEY = "soupy.dash.guild";
  // The bot rewrites data/bot_dashboard.json about every 15 seconds.
  const STATUS_POLL_MS = 20000;
  const NOTICE_LINGER_MS = 6000;

  /** "#database/profiles" as { tab, view }. Anything unrecognized falls back to the first. */
  function readRoute() {
    var parts = String(window.location.hash || "").replace(/^#/, "").split("/");
    return {
      tab: TABS.indexOf(parts[0]) !== -1 ? parts[0] : TABS[0],
      view: DB_VIEWS.indexOf(parts[1]) !== -1 ? parts[1] : DB_VIEWS[0],
    };
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

  function App() {
    const initialDataNode = document.getElementById("dashboard-initial");
    let initialStatus = { running: false, pid: null };
    try {
      if (initialDataNode) initialStatus = JSON.parse(initialDataNode.textContent || "{}").status || initialStatus;
    } catch (_err) {}

    const [route, setRoute] = useState(readRoute);
    const tab = route.tab;
    const [notice, setNotice] = useState("");
    const [noticeError, setNoticeError] = useState(false);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState(initialStatus);
    const [summary, setSummary] = useState(null);
    const [envVars, setEnvVars] = useState({});
    const [selectedCommand, setSelectedCommand] = useState(null);
    const [cmdEnvDrafts, setCmdEnvDrafts] = useState({});
    const [dashStatus, setDashStatus] = useState({});
    // The local Flux server, as the panel sees it (/api/flux/status). Empty
    // until the first answer, and when the panel predates that route.
    const [fluxStatus, setFluxStatus] = useState({});
    const [fluxBusy, setFluxBusy] = useState(false);
    const [fluxLog, setFluxLog] = useState(null);
    // Whether the card's log is open, for timers that outlive a render.
    const fluxLogOpenRef = useRef(false);
    const [activity, setActivity] = useState({});
    const [cmdChannelList, setCmdChannelList] = useState([]);
    const [cmdChannelConfig, setCmdChannelConfig] = useState({});
    const [logs, setLogs] = useState(["[loading logs...]"]);
    const [autoscroll, setAutoscroll] = useState(true);
    const [pauseLogs, setPauseLogs] = useState(false);
    const [logsConnState, setLogsConnState] = useState("connecting");
    const [logsRetryIn, setLogsRetryIn] = useState(0);
    const [logFilter, setLogFilter] = useState("");
    const [logLevels, setLogLevels] = useState({ DEBUG: true, INFO: true, WARNING: true, ERROR: true });
    const [dbStatus, setDbStatus] = useState({ databases: [] });
    const [dbGuild, setDbGuildState] = useState(function () {
      try {
        return localStorage.getItem(GUILD_STORAGE_KEY) || "";
      } catch (_err) {
        return "";
      }
    });
    const [runtimeFlags, setRuntimeFlags] = useState({ rag_enabled: false });
    const [ragStatus, setRagStatus] = useState(null);
    const [profileStatus, setProfileStatus] = useState(null);
    // Status bar restart confirm, and the Overview tab's collapsible "knob" sections
    const [restartConfirmOpen, setRestartConfirmOpen] = useState(false);
    const [overviewKnobs, setOverviewKnobs] = useState({ commands: false, rag: false, time: false });
    const [confirm, confirmDialog] = D.useConfirm();
    const logRef = useRef(null);
    const logsAttemptRef = useRef(0);
    const logsReconnectTimerRef = useRef(null);
    const logsCountdownTimerRef = useRef(null);
    // The socket's handler is made once, so it reads the pause switch through a ref.
    const pauseLogsRef = useRef(false);
    pauseLogsRef.current = pauseLogs;

    function notify(message, isError) {
      setNotice(message);
      setNoticeError(!!isError);
    }

    function navigate(nextTab, nextView) {
      var view = nextView || route.view;
      var hash = "#" + nextTab + (nextTab === "database" ? "/" + view : "");
      if (window.location.hash !== hash) window.history.pushState(null, "", hash);
      setRoute({ tab: nextTab, view: view });
    }

    function setTab(id) {
      navigate(id);
    }

    function setDbGuild(id) {
      setDbGuildState(id);
      try {
        localStorage.setItem(GUILD_STORAGE_KEY, id);
      } catch (_err) {}
    }

    const filteredLogs = useMemo(
      function () {
        const q = logFilter.trim().toLowerCase();
        const allLevelsOn = logLevels.DEBUG && logLevels.INFO && logLevels.WARNING && logLevels.ERROR;
        if (!q && allLevelsOn) return logs;
        const re = /\b(DEBUG|INFO|WARNING|WARN|ERROR)\b/i;
        return logs.filter(function (line) {
          if (q && line.toLowerCase().indexOf(q) === -1) return false;
          const m = line.match(re);
          if (!m) return true; // lines without a recognizable level are always shown
          var lvl = m[1].toUpperCase();
          if (lvl === "WARN") lvl = "WARNING";
          return logLevels[lvl] !== false;
        });
      },
      [logs, logFilter, logLevels]
    );

    useEffect(function () {
      loadSummary();
      loadEnv();
      loadRuntimeFlags();
      refreshStatus();
      loadEnvToggles();
      connectLogs();
      function onRoute() {
        setRoute(readRoute());
      }
      window.addEventListener("popstate", onRoute);
      window.addEventListener("hashchange", onRoute);
      // The status bar sticks just under the page nav, and the profile list
      // under both. Their heights follow the title, the zoom and line wraps,
      // so the stylesheet gets them measured.
      var rootEl = document.getElementById("dashboard-root");
      var nav = document.querySelector("nav.app-nav");
      var bar = rootEl.querySelector(".dash-sticky-bar");
      function measure() {
        rootEl.style.setProperty("--dash-nav-height", (nav ? nav.offsetHeight : 0) + "px");
        rootEl.style.setProperty("--dash-bar-height", (bar ? bar.offsetHeight : 0) + "px");
      }
      measure();
      var sizes = typeof ResizeObserver === "function" ? new ResizeObserver(measure) : null;
      if (sizes && nav) sizes.observe(nav);
      if (sizes && bar) sizes.observe(bar);
      // Keep the status bar current, but not from a background browser tab.
      var poll = setInterval(function () {
        if (document.visibilityState === "visible") refreshStatus(true);
      }, STATUS_POLL_MS);
      return function () {
        window.removeEventListener("popstate", onRoute);
        window.removeEventListener("hashchange", onRoute);
        if (sizes) sizes.disconnect();
        clearInterval(poll);
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(
      function () {
        if (tab === "database" || tab === "overview") {
          loadDatabases();
        }
        if (tab === "overview") {
          loadActivity();
        }
        if (tab === "stats") {
          loadSummary();
        }
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [tab]
    );

    // A success message clears itself; an error stays until it is dismissed.
    useEffect(
      function () {
        if (!notice || noticeError) return;
        var id = setTimeout(function () {
          setNotice("");
        }, NOTICE_LINGER_MS);
        return function () {
          clearTimeout(id);
        };
      },
      [notice, noticeError]
    );

    useEffect(
      function () {
        if ((tab !== "overview" && tab !== "database") || !dbGuild) {
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
        if (autoscroll && logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
      },
      [filteredLogs, autoscroll]
    );

    async function copyVisibleLogs() {
      if (await D.copyText(filteredLogs.join("\n"))) {
        notify("Copied " + filteredLogs.length + " log line(s) to clipboard.");
      } else {
        notify("Copy failed: the browser refused clipboard access.", true);
      }
    }

    const loadSummary = useCallback(async function () {
      try {
        const data = await readJson("/api/stats/summary?limit=12");
        setSummary(data);
      } catch (err) {
        setNotice("Failed to load stats: " + err.message);
        setNoticeError(true);
      }
    }, []);

    // `quiet` is the background poll: a failure there (the panel restarting,
    // a laptop waking up) isn't worth a message every 20 seconds.
    async function refreshStatus(quiet) {
      try {
        const data = await readJson("/api/bot/status");
        setStatus(data);
      } catch (err) {
        if (!quiet) notify("Failed to refresh status: " + err.message, true);
      }
      try {
        setDashStatus((await readJson("/api/bot/dashboard-status")) || {});
      } catch (_err) {
        // Extended status is a nicety; the bar still shows running/stopped.
      }
      try {
        setFluxStatus((await readJson("/api/flux/status")) || {});
      } catch (_err) {
        // A panel started before this route existed: the bot's own report is used.
      }
    }

    async function loadFluxLog() {
      try {
        setFluxLog(((await readJson("/api/flux/log?lines=60")) || {}).lines || []);
      } catch (err) {
        setFluxLog(["Could not read the Flux server's log: " + err.message]);
      }
    }

    async function fluxAction(action) {
      if (action !== "start") {
        const ok = await confirm({
          title: (action === "stop" ? "Stop" : "Restart") + " the Flux server?",
          message: "A picture being generated right now is lost. The model loads again on the next image.",
          confirmLabel: action === "stop" ? "Stop" : "Restart",
        });
        if (!ok) return;
      }
      setFluxBusy(true);
      try {
        const data = await readJson("/api/flux/" + action, { method: "POST" });
        setFluxStatus(data || {});
        notify(data.message || "Done", data.ok === false);
        // The server takes a few seconds to answer after a start. Don't leave
        // the card on "Starting…" until the next 20-second poll.
        [2500, 7000].forEach(function (ms) {
          setTimeout(function () {
            refreshStatus(true);
            if (fluxLogOpenRef.current) loadFluxLog();
          }, ms);
        });
      } catch (err) {
        notify("Flux server: " + err.message, true);
      } finally {
        setFluxBusy(false);
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
          // These two cards' pills read here too; without them they always loaded as off.
          SELF_MD_ENABLED: ["1", "true", "yes"].indexOf((vars.SELF_MD_ENABLED || "").toLowerCase()) !== -1,
          MUSING_ENABLED: ["1", "true", "yes", "y", "on"].indexOf((vars.MUSING_ENABLED || "").toLowerCase()) !== -1,
          // Defaults on when unset (matches the bot).
          USER_PROFILE_NIGHTLY_ENABLED:
            ["0", "false", "no", "off"].indexOf((vars.USER_PROFILE_NIGHTLY_ENABLED || "true").toLowerCase()) === -1,
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
        await D.postJson("/api/env/save", payload);
        var updated = Object.assign({}, envToggles);
        updated[key] = !current;
        setEnvToggles(updated);
      } catch (err) {
        notify("Could not save " + key + ": " + err.message, true);
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
      if (logsReconnectTimerRef.current) {
        clearTimeout(logsReconnectTimerRef.current);
        logsReconnectTimerRef.current = null;
      }
      if (logsCountdownTimerRef.current) {
        clearInterval(logsCountdownTimerRef.current);
        logsCountdownTimerRef.current = null;
      }
      setLogsRetryIn(0);
      setLogsConnState("connecting");
      try {
        const proto = window.location.protocol === "https:" ? "wss" : "ws";
        const ws = new WebSocket(proto + "://" + window.location.host + "/ws/logs");
        ws.onopen = function () {
          logsAttemptRef.current = 0;
          setLogsConnState("connected");
          pushLog("[logs connected]");
        };
        ws.onmessage = function (ev) {
          if (!pauseLogsRef.current) pushLog(String(ev.data || ""));
        };
        ws.onerror = function () {
          try { ws.close(); } catch (_) {}
        };
        ws.onclose = function () {
          const attempt = logsAttemptRef.current + 1;
          logsAttemptRef.current = attempt;
          // 1s, 2s, 4s, 8s, 16s, 30s, 30s, …
          const delay = Math.min(30, Math.pow(2, Math.min(attempt - 1, 5)));
          setLogsConnState("reconnecting");
          setLogsRetryIn(delay);
          pushLog("[logs disconnected — reconnecting in " + delay + "s]");
          logsCountdownTimerRef.current = setInterval(function () {
            setLogsRetryIn(function (curr) {
              if (curr <= 1) {
                clearInterval(logsCountdownTimerRef.current);
                logsCountdownTimerRef.current = null;
                return 0;
              }
              return curr - 1;
            });
          }, 1000);
          logsReconnectTimerRef.current = setTimeout(function () {
            logsReconnectTimerRef.current = null;
            connectLogs();
          }, delay * 1000);
        };
      } catch (_err) {
        setLogsConnState("unavailable");
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

    async function refreshSelfMemory() {
      if (!dbGuild) return;
      setBusy(true);
      setNotice("Queueing a refresh of Soupy's memory…");
      setNoticeError(false);
      try {
        await readJson("/api/self-memory/refresh/" + encodeURIComponent(dbGuild), { method: "POST" });
        setNotice(
          "Queued. The bot reads Soupy's new messages at its next chance (between members if a profile job is running). Progress shows under Database → Maintenance."
        );
      } catch (err) {
        setNotice("Could not queue the refresh: " + err.message);
        setNoticeError(true);
      } finally {
        setBusy(false);
      }
    }

    async function rebuildRagIndex() {
      if (!dbGuild) return;
      const ok = await confirm({
        title: "Rebuild the search index?",
        message: "Every archived message for this server is embedded again through LM Studio. That can take a while.",
        confirmLabel: "Rebuild",
      });
      if (!ok) return;
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

    async function loadEnv() {
      try {
        const data = await readJson("/api/env/get");
        const vars = data.vars || {};
        setEnvVars(vars);
      } catch (err) {
        setNotice("Failed to load environment: " + err.message);
        setNoticeError(true);
      }
    }

    async function loadDatabases() {
      try {
        const data = await readJson("/api/database/status");
        setDbStatus(data);
        // Keep the remembered server if it still has an archive; otherwise the biggest one.
        const ids = (data.databases || []).map(function (d) {
          return String(d.guild_id);
        });
        setDbGuildState(function (old) {
          return ids.indexOf(old) !== -1 ? old : ids[0] || "";
        });
      } catch (err) {
        notify("Failed to load databases: " + err.message, true);
      }
    }

    async function triggerScan() {
      if (!dbGuild) return;
      const ok = await confirm({
        title: "Scan this server now?",
        message: "The bot fetches messages posted since the last scan and adds them to the archive.",
        confirmLabel: "Scan now",
      });
      if (!ok) return;
      try {
        const data = await readJson("/api/database/scan/" + encodeURIComponent(dbGuild), { method: "POST" });
        notify(data.message || "Scan triggered");
        setTimeout(loadDatabases, 1200);
      } catch (err) {
        notify("Scan failed: " + err.message, true);
      }
    }

    const tabButton = function (id, label) {
      const isActive = tab === id;
      return e(
        "button",
        {
          type: "button",
          className: "dash-tab" + (isActive ? " dash-tab--active" : ""),
          "aria-current": isActive ? "page" : null,
          onClick: function () {
            setTab(id);
          },
        },
        label
      );
    };

    function formatUptime(s) {
      if (s == null) return "—";
      var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
      return (d > 0 ? d + "d " : "") + h + "h " + m + "m";
    }

    // The bot reports whether it can reach each backend. Until a freshly started
    // bot has reported, the answer is unknown, which is not the same as down.
    function reachClass(online) {
      return online == null ? "status-unknown" : online ? "status-on" : "status-off";
    }

    function reachTitle(online) {
      return online == null ? ": not reported yet" : online ? ": reachable" : ": unreachable";
    }

    // Flux is local, so the panel asks it directly: that answer is seconds old,
    // where the bot's report can be half a minute behind a Start or Stop.
    var fluxShown = !!(fluxStatus.enabled || dashStatus.flux_enabled);
    var fluxOnline = fluxStatus.reachable != null ? fluxStatus.reachable : dashStatus.flux_online;

    // Shown above every tab: whether the bot is up, and the buttons that change that.
    var statusBar = e(
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
      status.running && dashStatus.uptime_seconds != null
        ? e("div", { className: "dash-sticky-segment" },
            e("span", { className: "muted" }, "Uptime"),
            e("span", null, formatUptime(dashStatus.uptime_seconds))
          )
        : null,
      e("div", { className: "dash-sticky-segment", title: "LM Studio" + reachTitle(dashStatus.llm_online) },
        e("span", { className: "status-dot " + reachClass(dashStatus.llm_online) }),
        e("span", null, "LLM")
      ),
      fluxShown
        ? e("div", { className: "dash-sticky-segment", title: "Flux server" + reachTitle(fluxOnline) },
            e("span", { className: "status-dot " + reachClass(fluxOnline) }),
            e("span", null, "Flux")
          )
        : null,
      e("div", { className: "dash-sticky-segment", title: "Stable Diffusion backend" + reachTitle(dashStatus.sd_online) },
        e("span", { className: "status-dot " + reachClass(dashStatus.sd_online) }),
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
          title: "Reload status, stats and activity",
          onClick: function () { refreshStatus(); loadSummary(); loadEnvToggles(); loadActivity(); loadDatabases(); }
        }, "Refresh")
      )
    );

    return e(
      "section",
      { className: "dash-app" },
      e(
        // A <div>: Pico lays out a <nav> with space-between, which spreads the tabs apart.
        "div",
        { className: "dash-nav", role: "navigation", "aria-label": "Dashboard sections" },
        tabButton("overview", "Overview"),
        tabButton("model", "Model & Personality"),
        tabButton("stats", "Stats"),
        tabButton("media", "Media"),
        tabButton("database", "Database")
      ),
      statusBar,
      tab === "overview" && (function () {
        // A scheduled time as a countdown. Some timers report a phrase ("after
        // each scan") instead of a time; those pass through as they are.
        function timeUntil(iso) {
          if (!iso) return "—";
          if (typeof iso === "string" && isNaN(Date.parse(iso))) return iso;
          var diff = Math.max(0, Math.floor((new Date(iso).getTime() - Date.now()) / 60000));
          if (diff < 1) return "imminent";
          if (diff < 60) return "in " + diff + "m";
          if (diff < 1440) return "in " + Math.floor(diff / 60) + "h " + (diff % 60) + "m";
          return "in " + Math.floor(diff / 1440) + "d " + Math.floor((diff % 1440) / 60) + "h";
        }
        function timeAgo(iso) {
          if (!iso) return "never";
          var then = new Date(iso).getTime();
          if (isNaN(then)) return "—";
          var ago = Math.floor((Date.now() - then) / 60000);
          if (ago < 1) return "just now";
          if (ago < 60) return ago + "m ago";
          if (ago < 1440) return Math.floor(ago / 60) + "h " + (ago % 60) + "m ago";
          return Math.floor(ago / 1440) + "d ago";
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
            // Loops with nothing to trigger from here get no button.
            opts.runNowOnClick
              ? e("div", { className: "dash-loop-card-actions" },
                  e("button", {
                    className: "secondary",
                    disabled: !!opts.runNowDisabled,
                    title: opts.runNowTitle || null,
                    onClick: opts.runNowOnClick
                  }, opts.runNowLabel || "Run now")
                )
              : null
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

        var noGuildTitle = "Pick a server under RAG below";
        var selfMemory = ((dashStatus.self_memory || {})[dbGuild]) || {};

        // ---------- Band 1 — Live Activity ----------
        var bandLive = e(
          "article", { className: "dash-band-live" },
            e("div", { className: "logs-header" },
              e("h4", { style: { marginTop: 0, marginBottom: 0 } }, "Live Activity"),
              (function () {
                var label, mod;
                if (logsConnState === "connected") { label = "live"; mod = "logs-conn--on"; }
                else if (logsConnState === "reconnecting") {
                  label = logsRetryIn > 0 ? "reconnecting in " + logsRetryIn + "s" : "reconnecting…";
                  mod = "logs-conn--retry";
                }
                else if (logsConnState === "unavailable") { label = "unavailable"; mod = "logs-conn--off"; }
                else { label = "connecting…"; mod = "logs-conn--retry"; }
                return e("span", { className: "logs-conn " + mod, role: "status", "aria-live": "polite" },
                  e("span", { className: "logs-conn-dot" }), label,
                  logsConnState === "reconnecting" || logsConnState === "unavailable"
                    ? e("button", {
                        className: "secondary logs-conn-retry",
                        onClick: function () { logsAttemptRef.current = 0; connectLogs(); }
                      }, "Reconnect now")
                    : null
                );
              })()
            ),
            e("div", { className: "dash-controls logs-controls", style: { marginBottom: "0.4rem" } },
              e("button", { className: "secondary", onClick: function () { setPauseLogs(!pauseLogs); } },
                pauseLogs ? "Resume stream" : "Pause stream"),
              e("button", { className: "secondary", onClick: function () { setAutoscroll(!autoscroll); } },
                autoscroll ? "Auto-scroll on" : "Auto-scroll off"),
              e("button", { className: "secondary", onClick: function () { setLogs([]); } }, "Clear"),
              e("button", {
                className: "secondary",
                onClick: copyVisibleLogs,
                title: "Copy visible (filtered) log lines to clipboard"
              }, "Copy"),
              e("input", {
                type: "search",
                className: "logs-grep",
                placeholder: "Filter (substring)…",
                value: logFilter,
                onChange: function (ev) { setLogFilter(ev.target.value); }
              }),
              e("span", { className: "logs-level-set" },
                ["DEBUG", "INFO", "WARNING", "ERROR"].map(function (lvl) {
                  return e("label", { key: lvl, className: "logs-level logs-level--" + lvl.toLowerCase() },
                    e("input", {
                      type: "checkbox",
                      checked: logLevels[lvl] !== false,
                      onChange: function (ev) {
                        var v = ev.target.checked;
                        setLogLevels(function (curr) {
                          var next = Object.assign({}, curr);
                          next[lvl] = v;
                          return next;
                        });
                      }
                    }),
                    lvl
                  );
                })
              )
            ),
            (logFilter || !logLevels.DEBUG || !logLevels.INFO || !logLevels.WARNING || !logLevels.ERROR)
              ? e("div", { className: "logs-filter-summary muted" },
                  "Showing " + filteredLogs.length + " of " + logs.length + " line(s)")
              : null,
            e("div", { className: "dash-log mono", ref: logRef }, filteredLogs.join("\n"))
        );

        // ---------- Flux server card (joins the loops grid) ----------
        // Only with FLUX_ENABLED on. The panel can start and stop its own
        // server; one started by hand is shown and left alone.
        var fluxCard = null;
        if (fluxStatus.enabled) {
          var fluxState = fluxStatus.external ? "Running, started outside the panel"
            : fluxStatus.running && fluxStatus.reachable ? "Running"
            : fluxStatus.running ? "Starting…"
            // An exit nobody asked for: say how it ended. The reason is in the log.
            : fluxStatus.returncode && !fluxStatus.stopped_by_panel
              ? "Stopped on its own (exit code " + fluxStatus.returncode + ")"
            : "Stopped";
          var fluxModelLine = function (name, loaded) {
            return name ? name + (loaded ? " (loaded)" : " (loads on first use)") : "—";
          };
          var fluxMeta = [
            e("span", { key: "sl", className: "dash-loop-card-meta-label" }, "State:"),
            e("span", { key: "sv" }, fluxState)
          ];
          if (fluxStatus.reachable) {
            fluxMeta.push(e("span", { key: "ml", className: "dash-loop-card-meta-label" }, "Model:"));
            fluxMeta.push(e("span", { key: "mv" }, fluxModelLine(fluxStatus.model, fluxStatus.loaded)));
            fluxMeta.push(e("span", { key: "el", className: "dash-loop-card-meta-label" }, "Edit model:"));
            fluxMeta.push(e("span", { key: "ev" }, fluxModelLine(fluxStatus.edit_model, fluxStatus.edit_loaded)));
          }
          if (fluxStatus.running && fluxStatus.start_time) {
            fluxMeta.push(e("span", { key: "ul", className: "dash-loop-card-meta-label" }, "Started:"));
            fluxMeta.push(e("span", { key: "uv" }, timeAgo(fluxStatus.start_time)));
          }
          fluxCard = e("div", { key: "flux-server", className: "dash-loop-card" },
            e("div", { className: "dash-loop-card-header" },
              e("h5", { className: "dash-loop-card-title" }, "Flux server"),
              e("span", { className: "status-dot " + reachClass(fluxStatus.reachable), title: "Flux server" + reachTitle(fluxStatus.reachable) })
            ),
            e("div", { className: "muted", style: { fontSize: "0.76rem" } },
              fluxStatus.external
                ? "Stop it in the terminal it was started in. The panel only stops a server it started."
                : fluxStatus.autostart
                  ? "Starts with the panel (FLUX_SERVER_AUTOSTART)."
                  : "Set FLUX_SERVER_AUTOSTART=true to start it with the panel."),
            e("div", { className: "dash-loop-card-meta" }, fluxMeta),
            e("div", { className: "dash-loop-card-actions" },
              e("button", {
                disabled: fluxBusy || fluxStatus.running || fluxStatus.reachable,
                onClick: function () { fluxAction("start"); }
              }, "Start"),
              e("button", {
                className: "secondary",
                disabled: fluxBusy || !fluxStatus.running,
                onClick: function () { fluxAction("stop"); }
              }, "Stop"),
              e("button", {
                className: "secondary",
                disabled: fluxBusy || !fluxStatus.running,
                title: "Picks up a change to FLUX_* settings or to flux_server.py",
                onClick: function () { fluxAction("restart"); }
              }, "Restart"),
              e("button", {
                className: "secondary",
                title: "The server's output since the panel started. The whole log is logs/flux_server.log.",
                onClick: function () {
                  fluxLogOpenRef.current = !fluxLog;
                  if (fluxLog) setFluxLog(null); else loadFluxLog();
                }
              }, fluxLog ? "Hide log" : "Show log")
            ),
            fluxLog
              ? e("div", {
                  className: "dash-log mono",
                  style: { maxHeight: "14rem", marginTop: "0.5rem", fontSize: "0.72rem" },
                  // Newest lines are at the bottom: go there when the log changes.
                  ref: function (el) {
                    var mark = fluxLog.length + ":" + (fluxLog[fluxLog.length - 1] || "");
                    if (el && el.getAttribute("data-shown") !== mark) {
                      el.setAttribute("data-shown", mark);
                      el.scrollTop = el.scrollHeight;
                    }
                  }
                },
                  fluxLog.length ? fluxLog.join("\n") : "Nothing yet. The panel only has the output of a server it started.")
              : null
          );
        }

        // ---------- Band 2 — Loops grid ----------
        var timers = dashStatus.timers || {};
        // The musings loop fires at most once a day, so "what happened today"
        // is more useful on the card than a count. Keys come from
        // last_handled_status in soupy/cogs/musings.py.
        var MUSING_STATUS_LABELS = {
          posted: "Posted today",
          no_thought_generated: "Ran today, produced nothing",
          skipped_past_window: "Skipped — window had closed",
          channel_unavailable: "Skipped — channel unreachable",
          not_a_guild_channel: "Skipped — channel is not in a guild",
          error: "Failed today — see logs"
        };
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
            todayLine: dailyPostsToday + " / 2 posts"
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
            todayLine: D.plural(bskyAct.replies_today || 0, "reply", "replies") + " · "
                     + D.plural(bskyAct.posts_today || 0, "post") + " · "
                     + D.plural(bskyAct.reposts_today || 0, "repost")
          }),
          loopCard({
            key: "loop-archive-scan",
            title: "Archive Scan",
            on: null,
            nextRunIso: (timers.archive_scan || {}).next_run,
            lastRunIso: (timers.archive_scan || {}).last_run,
            intervalLabel: (timers.archive_scan || {}).interval || null,
            todayLine: null,
            runNowDisabled: !dbGuild || !status.running || busy,
            runNowOnClick: triggerScan,
            runNowTitle: !dbGuild ? noGuildTitle : !status.running ? "Start the bot first" : null
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
            runNowTitle: !dbGuild ? noGuildTitle : null
          }),
          loopCard({
            key: "loop-self-memory",
            title: "Soupy's Memory",
            on: !!envToggles.SELF_MD_ENABLED,
            onToggle: function () { toggleEnvVar("SELF_MD_ENABLED"); },
            toggleHint: "Soupy's memory of itself, built from its own messages at the end of each profile refresh. Toggling writes .env-stable; takes effect on next bot restart.",
            nextRunIso: (timers.profile_nightly || {}).next_run,
            lastRunIso: selfMemory.updated_at || null,
            intervalLabel: "after each profile refresh",
            todayLine: !dbGuild
              ? null
              : selfMemory.items != null
                ? fmtNum(selfMemory.items) + " memories"
                : "not built yet",
            runNowDisabled: !dbGuild || busy,
            runNowOnClick: refreshSelfMemory,
            runNowTitle: !dbGuild ? noGuildTitle : "Queue a refresh; the bot runs it between profile members"
          }),
          loopCard({
            key: "loop-profile-nightly",
            title: "Profile Refresh",
            on: !!envToggles.USER_PROFILE_NIGHTLY_ENABLED,
            onToggle: function () { toggleEnvVar("USER_PROFILE_NIGHTLY_ENABLED"); },
            toggleHint: "Nightly update of member profiles from new messages. Toggling writes .env-stable; takes effect on next bot restart.",
            nextRunIso: (timers.profile_nightly || {}).next_run,
            lastRunIso: (timers.profile_nightly || {}).last_run,
            intervalLabel: (timers.profile_nightly || {}).interval || null,
            todayLine: null,
            runNowDisabled: !dbGuild,
            runNowLabel: "Open the profile job →",
            runNowOnClick: function () { navigate("database", "maintenance"); },
            runNowTitle: !dbGuild ? noGuildTitle : null
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
            todayLine: MUSING_STATUS_LABELS[(timers.musings || {}).last_status] || null
          }),
          fluxCard
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
            e("button", { className: "secondary", onClick: function () { setTab("stats"); } }, "All stats →")
          )
        );

        // ---------- Band 4 — Knobs accordion ----------
        // time_awareness is on unless the flags file says false, so "not loaded yet" reads as on.
        var timeKnobBody = e("div", null,
          e("p", { className: "muted", style: { marginTop: 0 } },
            "When enabled, Soupy knows the current date and time (TIMEZONE) in chat, and how long ago older ",
            "chat history and memories were. On by default; takes effect on the next reply."),
          e("label", { style: { display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" } },
            e("input", {
              type: "checkbox",
              checked: runtimeFlags.time_awareness !== false,
              disabled: busy,
              onChange: function (ev) {
                var v = !!ev.target.checked;
                setBusy(true);
                setNoticeError(false);
                readJson("/api/runtime-flags", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ time_awareness: v })
                })
                  .then(function (data) {
                    setRuntimeFlags(data || Object.assign({}, runtimeFlags, { time_awareness: v }));
                    setNotice(v ? "Time awareness enabled." : "Time awareness disabled.");
                  })
                  .catch(function (err) { setNotice("Failed to save: " + err.message); setNoticeError(true); })
                  .finally(function () { setBusy(false); });
              }
            }),
            e("span", null, "Enable time awareness in chat replies")
          )
        );

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
            e("label", null, "Server"),
            e("select", {
              value: dbGuild,
              onChange: function (ev) { setDbGuild(ev.target.value); }
            },
              (dbStatus.databases || []).length
                ? (dbStatus.databases || []).map(function (d) {
                    return e("option", { key: d.guild_id, value: d.guild_id }, d.guild_name || "Server " + d.guild_id);
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
            : e("p", { className: "muted" }, "Pick a server to see its index."),
          profileStatus && profileStatus.ok
            ? e("p", { className: "muted", style: { marginBottom: "0.5rem" } },
                "Stored member profiles: ",
                e("span", { className: "mono" }, fmtNum(profileStatus.profile_count || 0)),
                profileStatus.latest_updated_at
                  ? e("span", null, " · Last profile update: ", D.timeAgo(profileStatus.latest_updated_at))
                  : null)
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
              onClick: function () { navigate("database", "profiles"); }
            }, "Read the profiles →")
          ),
          e("p", { className: "muted", style: { marginTop: "0.5rem", fontSize: "0.88rem" } },
            "Set ", e("span", { className: "mono" }, "RAG_EMBEDDING_MODEL"),
            " in ", e("span", { className: "mono" }, ".env-stable"),
            " to the LM Studio embedding model id. Load that embedding model in LM Studio; ",
            e("span", { className: "mono" }, "OPENAI_BASE_URL"),
            " should point at your server. Profile batch uses ",
            e("span", { className: "mono" }, "LOCAL_CHAT"), ", ",
            e("span", { className: "mono" }, "USER_PROFILES_BATCH_MAX_USERS"), ", ",
            e("span", { className: "mono" }, "USER_PROFILE_PASS_MAX_MESSAGES"),
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
                keys: ["BEHAVIOUR_SEARCH", "SEARCH_SELECT_TEMPERATURE", "SEARCH_SUMMARY_TEMPERATURE", "SEARCH_BACKEND_TIMEOUT_SECONDS", "SEARCH_RESULTS_PER_QUERY", "SEARCH_BACKENDS", "URL_FETCH_TIMEOUT", "URL_MAX_CONTENT_LENGTH", "URL_CACHE_TTL_SECONDS", "LOCAL_CHAT"],
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
                keys: ["BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD", "BLUESKY_AUTO_REPLY", "BLUESKY_REPLIES_MIN", "BLUESKY_REPLIES_MAX", "BLUESKY_REPOSTS_PER_DAY", "BLUESKY_POSTS_PER_DAY", "BLUESKY_MIN_GAP_MINUTES", "BLUESKY_MAX_LIKES_PER_DAY", "BLUESKY_MAX_FOLLOWS_PER_DAY", "BLUESKY_ARTICLE_FRESHNESS_DAYS", "BLUESKY_TOPIC_DEDUP_SIM", "BLUESKY_TOPIC_DEDUP_DAYS"],
              },
              soupyself: {
                desc: "Soupy's memory of itself — built from its own messages with the profile refresh",
                keys: ["SELF_MD_ENABLED", "SELF_MD_ANCHOR_MAX_CHARS", "SELF_MD_ANCHOR_FALLBACK_CHARS", "RAG_SELF_KNOWLEDGE_MIN_SIM", "RAG_SELF_KNOWLEDGE_MAX_CHARS", "RAG_SELF_KNOWLEDGE_TOP_K"],
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
                keys: ["MUSING_ENABLED", "MUSING_CHANNEL_ID", "MUSING_HOUR_MIN", "MUSING_HOUR_MAX"],
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
              SEARCH_BACKEND_TIMEOUT_SECONDS: "12",
              SEARCH_RESULTS_PER_QUERY: "5",
              SEARCH_BACKENDS: "yahoo,yandex,mojeek,brave,duckduckgo",
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
              MUSING_HOUR_MIN: "6",
              MUSING_HOUR_MAX: "20",
              SELF_MD_ENABLED: "false",
              SELF_MD_ANCHOR_MAX_CHARS: "600",
              SELF_MD_ANCHOR_FALLBACK_CHARS: "600",
              RAG_SELF_KNOWLEDGE_MIN_SIM: "0.3",
              RAG_SELF_KNOWLEDGE_MAX_CHARS: "2500",
              RAG_SELF_KNOWLEDGE_TOP_K: "5",
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
              className: "dash-plain dash-knob-header",
              role: "button",
              tabIndex: 0,
              "aria-expanded": overviewKnobs.commands ? "true" : "false",
              onClick: knobToggle("commands"),
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); knobToggle("commands")(); } }
            },
              e("span", null, "Slash Commands"),
              e("span", { className: "dash-knob-chevron", "aria-hidden": "true" }, "›")
            ),
            overviewKnobs.commands
              ? e("div", { className: "dash-knob-body" }, slashCommandsBody)
              : null
          ),
          e("div", { className: "dash-knob" },
            e("div", {
              className: "dash-plain dash-knob-header",
              role: "button",
              tabIndex: 0,
              "aria-expanded": overviewKnobs.rag ? "true" : "false",
              onClick: knobToggle("rag"),
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); knobToggle("rag")(); } }
            },
              e("span", null, "RAG (archived messages)"),
              e("span", { className: "dash-knob-chevron", "aria-hidden": "true" }, "›")
            ),
            overviewKnobs.rag
              ? e("div", { className: "dash-knob-body" }, ragKnobBody)
              : null
          ),
          e("div", { className: "dash-knob" },
            e("div", {
              className: "dash-plain dash-knob-header",
              role: "button",
              tabIndex: 0,
              "aria-expanded": overviewKnobs.time ? "true" : "false",
              onClick: knobToggle("time"),
              onKeyDown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); knobToggle("time")(); } }
            },
              e("span", null, "Time awareness"),
              e("span", { className: "dash-knob-chevron", "aria-hidden": "true" }, "›")
            ),
            overviewKnobs.time
              ? e("div", { className: "dash-knob-body" }, timeKnobBody)
              : null
          ),
        );

        return e(
          React.Fragment,
          null,
          bandLive,
          bandLoops,
          bandSnapshot,
          bandKnobs
        );
      })(),
      e(D.ModelTab, {
        active: tab === "model",
        botRunning: !!status.running,
        botStartedAt: status.start_time,
        botModel: dashStatus.model,
        busy: busy,
        onBusy: setBusy,
        notify: notify,
        confirm: confirm,
        onStatus: setStatus,
        onRefreshStatus: refreshStatus,
      }),
      tab === "stats" &&
        e(StatsStudio, {
          summary: summary,
          active: tab === "stats",
          onReload: loadSummary,
        }),
      e(D.MediaTab, { active: tab === "media" }),
      e(D.DatabaseTab, {
        active: tab === "database",
        guild: dbGuild,
        onGuild: setDbGuild,
        databases: dbStatus.databases || [],
        onReloadDatabases: loadDatabases,
        view: route.view,
        onView: function (view) {
          navigate("database", view);
        },
        botRunning: !!status.running,
        busy: busy,
        notify: notify,
        confirm: confirm,
        ragStatus: ragStatus,
        onRebuildRag: rebuildRagIndex,
        onScan: triggerScan,
      }),
      e(D.Toast, {
        message: notice,
        error: noticeError,
        onDismiss: function () {
          setNotice("");
        },
      }),
      confirmDialog
    );
  }

  const root = document.getElementById("dashboard-root");
  if (root) {
    ReactDOM.createRoot(root).render(e(App));
  }
})();
