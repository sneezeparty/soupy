/*
 * Shared pieces for the dashboard scripts: fetch + formatting helpers and the
 * overlay primitives (modal, confirm dialog, toast). Loaded first; the tab
 * scripts and dashboard-app.js read everything off window.SoupyDash.
 */
(function () {
  const e = React.createElement;
  const { useCallback, useEffect, useRef, useState } = React;
  const D = (window.SoupyDash = window.SoupyDash || {});

  function fmtNum(v) {
    return Number(v || 0).toLocaleString();
  }

  function plural(n, one, many) {
    return fmtNum(n) + " " + (Number(n) === 1 ? one : many || one + "s");
  }

  function fmtBytes(n) {
    var v = Number(n || 0);
    if (v < 1024) return v + " B";
    if (v < 1024 * 1024) return (v / 1024).toFixed(0) + " KB";
    if (v < 1024 * 1024 * 1024) return (v / (1024 * 1024)).toFixed(1) + " MB";
    return (v / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }

  async function readJson(url, init) {
    const res = await fetch(url, init);
    let data = null;
    try {
      data = await res.json();
    } catch (_err) {
      data = null;
    }
    if (!res.ok) {
      throw new Error((data && (data.message || data.error)) || "Request failed (" + res.status + ")");
    }
    return data || {};
  }

  function postJson(url, body) {
    return readJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  function previewText(text, maxLen) {
    var s = String(text || "").replace(/\s+/g, " ").trim();
    if (s.length <= maxLen) return s;
    return s.slice(0, Math.max(0, maxLen - 1)) + "…";
  }

  // ---- Time -----------------------------------------------------------------
  // The archive stores UTC, mostly without a zone suffix. Everything on screen
  // is the browser's local time.

  function parseUtc(raw) {
    if (!raw) return null;
    var s = String(raw).trim().replace(" ", "T");
    if (!/(Z|[+-]\d\d:?\d\d)$/.test(s)) s += "Z";
    var d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }

  function fmtDateTime(raw) {
    var d = raw instanceof Date ? raw : parseUtc(raw);
    if (!d) return "";
    var opts = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
    if (d.getFullYear() !== new Date().getFullYear()) opts.year = "numeric";
    return d.toLocaleString(undefined, opts);
  }

  function fmtDateTimeFull(raw) {
    var d = raw instanceof Date ? raw : parseUtc(raw);
    if (!d) return "";
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  /** "2023-03-03" → "Mar 2023". A calendar day, so no zone shift. */
  function fmtMonth(day) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(day || ""));
    if (!m) return "";
    return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])).toLocaleDateString(undefined, {
      month: "short",
      year: "numeric",
    });
  }

  function timeAgo(raw) {
    var d = raw instanceof Date ? raw : parseUtc(raw);
    if (!d) return "never";
    var mins = Math.floor((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return mins + "m ago";
    var hours = Math.floor(mins / 60);
    if (hours < 24) return hours + "h " + (mins % 60) + "m ago";
    var days = Math.floor(hours / 24);
    if (days < 60) return days + "d ago";
    return fmtDateTime(d);
  }

  /** A local Date as the archive's UTC "YYYY-MM-DD HH:MM:SS". */
  function utcStamp(d) {
    return d.toISOString().slice(0, 19).replace("T", " ");
  }

  async function copyText(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (_err) {
      // Clipboard API is refused on plain-http LAN addresses; use the old way.
    }
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (_err) {
      return false;
    }
  }

  function useDebounced(value, delayMs) {
    const [debounced, setDebounced] = useState(value);
    useEffect(
      function () {
        var id = setTimeout(function () {
          setDebounced(value);
        }, delayMs);
        return function () {
          clearTimeout(id);
        };
      },
      [value, delayMs]
    );
    return debounced;
  }

  /** Split text into strings and <a> elements, so URLs in archived messages are clickable. */
  function linkify(text) {
    var s = String(text || "");
    var parts = [];
    var re = /https?:\/\/[^\s<>\]]+/g;
    var last = 0;
    var m;
    while ((m = re.exec(s))) {
      var url = m[0].replace(/[),.;!?]+$/, "");
      if (m.index > last) parts.push(s.slice(last, m.index));
      parts.push(e("a", { key: "l" + m.index, href: url, target: "_blank", rel: "noreferrer" }, url));
      last = m.index + url.length;
      re.lastIndex = last;
    }
    if (last < s.length) parts.push(s.slice(last));
    return parts;
  }

  // ---- Small components -----------------------------------------------------

  /**
   * Uniform loading/error/empty/info presentation.
   * Usage: PanelState({ kind: "loading"|"error"|"empty"|"info", message: "...", inline: true })
   */
  function PanelState(opts) {
    var o = opts || {};
    var kind = o.kind || "info";
    var cls = "panel-state panel-state--" + kind + (o.inline ? " panel-state--inline" : "");
    var icon = null;
    if (kind === "loading") icon = e("span", { className: "panel-state-spinner", "aria-hidden": "true" });
    else if (kind === "error") icon = e("span", { className: "panel-state-glyph", "aria-hidden": "true" }, "!");
    else if (kind === "empty") icon = e("span", { className: "panel-state-glyph", "aria-hidden": "true" }, "∅");
    else if (kind === "info") icon = e("span", { className: "panel-state-glyph", "aria-hidden": "true" }, "i");
    return e(
      "div",
      { className: cls, role: kind === "error" ? "alert" : "status", "aria-live": "polite" },
      icon,
      e("span", { className: "panel-state-text" }, o.message || ""),
      o.children ? e("span", { className: "panel-state-extra" }, o.children) : null
    );
  }

  /** A row of mutually exclusive buttons. options: [{ value, label, count?, dot? }] */
  function Segmented({ value, options, onChange, label }) {
    return e(
      "div",
      { className: "dash-seg", role: "radiogroup", "aria-label": label || null },
      options.map(function (opt) {
        var on = opt.value === value;
        return e(
          "button",
          {
            key: opt.value,
            type: "button",
            className: "dash-seg-btn" + (on ? " dash-seg-btn--on" : ""),
            role: "radio",
            "aria-checked": on ? "true" : "false",
            onClick: function () {
              if (!on) onChange(opt.value);
            },
          },
          opt.label,
          opt.count != null ? e("span", { className: "dash-seg-count" }, fmtNum(opt.count)) : null,
          opt.dot ? e("span", { className: "dash-seg-dot", title: opt.dot }) : null
        );
      })
    );
  }

  /** "1–50 of 14,115" with previous/next buttons. */
  function Pager({ offset, limit, total, onPage, disabled }) {
    var start = total ? offset + 1 : 0;
    var end = Math.min(offset + limit, total);
    return e(
      "div",
      { className: "dash-pager" },
      e("span", { className: "dash-pager-range" }, fmtNum(start) + "–" + fmtNum(end) + " of " + fmtNum(total)),
      e(
        "button",
        {
          type: "button",
          className: "secondary",
          disabled: disabled || offset <= 0,
          onClick: function () {
            onPage(Math.max(0, offset - limit));
          },
        },
        "‹ Prev"
      ),
      e(
        "button",
        {
          type: "button",
          className: "secondary",
          disabled: disabled || offset + limit >= total,
          onClick: function () {
            onPage(offset + limit);
          },
        },
        "Next ›"
      )
    );
  }

  // ---- Overlays -------------------------------------------------------------

  function overlayRoot() {
    var el = document.getElementById("dashboard-overlays");
    if (!el) {
      el = document.createElement("div");
      el.id = "dashboard-overlays";
      document.body.appendChild(el);
    }
    return el;
  }

  var overlayStack = [];

  function trapTab(ev, container) {
    if (!container) return;
    var nodes = container.querySelectorAll(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    if (!nodes.length) {
      ev.preventDefault();
      return;
    }
    var first = nodes[0];
    var last = nodes[nodes.length - 1];
    var active = document.activeElement;
    if (!container.contains(active)) {
      ev.preventDefault();
      first.focus();
    } else if (ev.shiftKey && active === first) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && active === last) {
      ev.preventDefault();
      first.focus();
    }
  }

  /**
   * A full-viewport layer over the page. Esc and a click on the backdrop call
   * onClose; Tab stays inside; the page behind doesn't scroll. Keys are read
   * at the document so they still arrive when focus has fallen back to <body>
   * (a focused button that just got disabled). onKey gets every other key.
   */
  function Overlay(props) {
    const ref = useRef(null);
    const latest = useRef(props);
    latest.current = props;

    useEffect(function () {
      var token = {};
      overlayStack.push(token);
      var previousFocus = document.activeElement;
      var previousOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      var target = ref.current && (ref.current.querySelector("[data-autofocus]") || ref.current);
      if (target) target.focus();

      function onKey(ev) {
        if (overlayStack[overlayStack.length - 1] !== token) return;
        var p = latest.current;
        if (ev.key === "Escape") {
          ev.preventDefault();
          p.onClose();
        } else if (ev.key === "Tab") {
          trapTab(ev, ref.current);
        } else if (p.onKey) {
          p.onKey(ev);
        }
      }
      document.addEventListener("keydown", onKey);
      return function () {
        document.removeEventListener("keydown", onKey);
        overlayStack.splice(overlayStack.indexOf(token), 1);
        document.body.style.overflow = previousOverflow;
        if (previousFocus && previousFocus.focus && document.contains(previousFocus)) previousFocus.focus();
      };
    }, []);

    return ReactDOM.createPortal(
      e(
        "div",
        {
          ref: ref,
          className: "dash-overlay " + (props.className || ""),
          role: "dialog",
          "aria-modal": "true",
          "aria-label": props.label || null,
          tabIndex: -1,
          onMouseDown: function (ev) {
            if (ev.target === ev.currentTarget) props.onClose();
          },
        },
        props.children
      ),
      overlayRoot()
    );
  }

  function ConfirmDialog({ request, onAnswer }) {
    return e(
      Overlay,
      {
        className: "dash-overlay--center",
        label: request.title || "Confirm",
        onClose: function () {
          onAnswer(false);
        },
      },
      e(
        "div",
        { className: "dash-dialog" },
        request.title ? e("h4", { className: "dash-dialog-title" }, request.title) : null,
        request.message ? e("p", { className: "dash-dialog-body" }, request.message) : null,
        // What will happen, one line each. { text, warn: true } marks a caution.
        request.details && request.details.length
          ? e(
              "ul",
              { className: "dash-dialog-list" },
              request.details.map(function (line, i) {
                var warn = typeof line === "object" && line.warn;
                return e(
                  "li",
                  { key: i, className: warn ? "dash-dialog-warn" : null },
                  typeof line === "object" ? line.text : line
                );
              })
            )
          : null,
        e(
          "div",
          { className: "dash-dialog-actions" },
          // Enter lands on the focused button: the safe one when the action destroys something.
          e(
            "button",
            {
              type: "button",
              className: "secondary",
              "data-autofocus": request.danger ? "true" : null,
              onClick: function () {
                onAnswer(false);
              },
            },
            request.cancelLabel || "Cancel"
          ),
          e(
            "button",
            {
              type: "button",
              className: request.danger ? "dash-btn-danger" : "",
              "data-autofocus": request.danger ? null : "true",
              onClick: function () {
                onAnswer(true);
              },
            },
            request.confirmLabel || "OK"
          )
        )
      )
    );
  }

  /**
   * In-page replacement for window.confirm.
   * const [confirm, dialog] = useConfirm();  render `dialog`;  if (await confirm({ title, message, details })) ...
   */
  function useConfirm() {
    const [request, setRequest] = useState(null);
    const ask = useCallback(function (opts) {
      return new Promise(function (resolve) {
        setRequest(Object.assign({}, opts, { resolve: resolve }));
      });
    }, []);
    const dialog = request
      ? e(ConfirmDialog, {
          request: request,
          onAnswer: function (ok) {
            request.resolve(ok);
            setRequest(null);
          },
        })
      : null;
    return [ask, dialog];
  }

  /** The one status message, pinned to the corner so it shows wherever the page is scrolled. */
  function Toast({ message, error, onDismiss }) {
    if (!message) return null;
    return ReactDOM.createPortal(
      e(
        "div",
        { className: "dash-toast" + (error ? " dash-toast--error" : ""), role: error ? "alert" : "status" },
        e("span", { className: "dash-toast-text" }, message),
        e(
          "button",
          { type: "button", className: "dash-toast-close", "aria-label": "Dismiss", onClick: onDismiss },
          "×"
        )
      ),
      overlayRoot()
    );
  }

  Object.assign(D, {
    e: e,
    fmtNum: fmtNum,
    plural: plural,
    fmtBytes: fmtBytes,
    readJson: readJson,
    postJson: postJson,
    previewText: previewText,
    parseUtc: parseUtc,
    fmtDateTime: fmtDateTime,
    fmtDateTimeFull: fmtDateTimeFull,
    fmtMonth: fmtMonth,
    timeAgo: timeAgo,
    utcStamp: utcStamp,
    copyText: copyText,
    useDebounced: useDebounced,
    linkify: linkify,
    PanelState: PanelState,
    Segmented: Segmented,
    Pager: Pager,
    Overlay: Overlay,
    useConfirm: useConfirm,
    Toast: Toast,
  });
})();
