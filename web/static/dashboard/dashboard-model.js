/*
 * The Model & Personality tab: which model writes Soupy's replies, and the
 * prompt files that tell it how. Reads window.SoupyDash (dashboard-core.js).
 */
(function () {
  const D = window.SoupyDash;
  const e = D.e;
  const fmtNum = D.fmtNum;
  const { useCallback, useEffect, useMemo, useRef, useState } = React;

  // Both mirror the bot: lm_studio.HEADROOM, and estimate_tokens() in the chat path.
  const HEADROOM = 0.85;
  const CHARS_PER_TOKEN = 3.5;

  // Page code is read from disk on every load; server code only when the panel
  // starts. After an update the two disagree until the panel is restarted.
  const OLD_SERVER = "The panel is still running its previous server code. Restart it (run_all.py) to use this tab.";

  function gb(bytes) {
    var v = Number(bytes || 0) / (1024 * 1024 * 1024);
    if (!v) return "";
    return (v >= 10 ? v.toFixed(0) : v.toFixed(1)) + " GB";
  }

  function tokens(n) {
    return fmtNum(n) + " tokens";
  }

  function percent(part, whole) {
    return Math.round((100 * part) / whole) + "%";
  }

  /** The model a configured name refers to: its key, or the id of one of its loaded copies. */
  function findModel(models, name) {
    if (!name) return null;
    return (
      models.find(function (m) {
        return m.key === name;
      }) ||
      models.find(function (m) {
        return (m.instances || []).some(function (i) {
          return i.id === name;
        });
      }) ||
      null
    );
  }

  function Note({ kind, children }) {
    return e("p", { className: "model-note model-note--" + (kind || "info") }, React.Children.toArray(children));
  }

  /** Restart the bot and return its status. A bot that doesn't come back is an error. */
  async function restartBot() {
    const status = await D.postJson("/api/bot/restart");
    if (status.ok === false) throw new Error(status.message || "the bot did not start");
    return status;
  }

  const ROLE_LABELS = { chat: "chat", vision: "images", embeddings: "embeddings" };

  function RoleChips({ roles }) {
    if (!roles || !roles.length) return e("span", { className: "muted" }, "not used by Soupy");
    return roles.map(function (role) {
      return e("span", { key: role, className: "model-role model-role--" + role }, ROLE_LABELS[role] || role);
    });
  }

  // ---- Chat model -----------------------------------------------------------

  /** What a plan from the server will do, as the lines of the confirm dialog. */
  function planLines(plan) {
    var lines = [];
    (plan.unload || []).forEach(function (u) {
      lines.push("Unload " + u.id + (u.reason ? " (" + u.reason + ")" : "") + ".");
    });
    if (plan.load) {
      var size = gb(plan.load.size_bytes);
      lines.push("Load " + plan.load.key + " at " + tokens(plan.load.context_length) + (size ? " (" + size + ")" : "") + ".");
    }
    var staying = (plan.keep || []).filter(function (k) {
      return k.key !== plan.model;
    });
    if (staying.length && (plan.load || (plan.unload || []).length)) {
      lines.push(
        "Stays loaded: " +
          staying
            .map(function (k) {
              var roles = (k.roles || []).map(function (r) {
                return ROLE_LABELS[r] || r;
              });
              return k.id + (roles.length ? " for " + roles.join(" and ") : "");
            })
            .join(", ") +
          "."
      );
    }
    if (plan.native && plan.loaded_bytes_after && (plan.load || (plan.unload || []).length)) {
      lines.push("LM Studio then holds about " + gb(plan.loaded_bytes_after) + " of models.");
    }
    var keys = Object.keys(plan.env || {}).sort();
    if (keys.length) {
      lines.push(
        "Save to .env-stable: " +
          keys
            .map(function (k) {
              return k + "=" + plan.env[k];
            })
            .join(", ") +
          "."
      );
    }
    lines.push(
      plan.bot_running
        ? "The bot stops while this runs and starts again after. A reply or profile job in progress is cut off."
        : "The bot is stopped and stays stopped."
    );
    (plan.warnings || []).forEach(function (w) {
      lines.push({ text: w, warn: true });
    });
    return lines;
  }

  function ModelCard({ data, loading, onReload, botRunning, botModel, busy, onBusy, notify, confirm, onChanged }) {
    const settings = (data && data.settings) || {};
    const models = (data && data.models) || [];
    const native = !!(data && data.native);
    const current = findModel(models, settings.chat);
    const currentKey = current ? current.key : settings.chat || "";
    const loadedContext = current && current.loaded ? current.loaded_context : null;

    const [model, setModel] = useState("");
    const [context, setContext] = useState("");
    const [budget, setBudget] = useState("");
    const [visionFollows, setVisionFollows] = useState(false);
    const [working, setWorking] = useState("");
    const [formError, setFormError] = useState("");
    // The form follows the server until the user changes a field.
    const [edited, setEdited] = useState(false);

    const choices = useMemo(
      function () {
        var rows = models
          .filter(function (m) {
            return m.type !== "embedding";
          })
          .map(function (m) {
            var bits = [m.key, m.quant, gb(m.size_bytes), m.loaded ? "loaded" : ""].filter(Boolean);
            return { value: m.key, label: bits.join(" · ") };
          });
        var known = rows.map(function (r) {
          return r.value;
        });
        // Without a list from the server, offer the names saved in AVAILABLE_MODELS.
        if (!rows.length) {
          (settings.saved_list || []).forEach(function (name) {
            if (known.indexOf(name) === -1) {
              rows.push({ value: name, label: name });
              known.push(name);
            }
          });
        }
        if (currentKey && known.indexOf(currentKey) === -1) {
          rows.unshift({ value: currentKey, label: currentKey + (models.length ? " · not on the server" : "") });
        }
        return rows;
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [data]
    );

    function defaultContext(key) {
      var target = findModel(models, key);
      if (target && target.loaded && target.loaded_context) return target.loaded_context;
      var guess = loadedContext || Math.ceil((settings.chat_budget || 16000) / HEADROOM / 1024) * 1024;
      if (target && target.max_context) guess = Math.min(guess, target.max_context);
      return guess;
    }

    function defaultVisionFollows(key) {
      var target = findModel(models, key);
      // Follow when the chat model has been reading images and the new one can.
      return !!settings.vision_enabled && settings.vision === currentKey && !(target && target.vision === false);
    }

    useEffect(
      function () {
        if (!data || edited) return;
        setModel(currentKey);
        setContext(native ? String(defaultContext(currentKey)) : "");
        setBudget(String(settings.chat_budget || ""));
        setVisionFollows(false);
        setFormError("");
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [data, edited]
    );

    function pickModel(key) {
      setEdited(true);
      setFormError("");
      setModel(key);
      if (native) setContext(String(defaultContext(key)));
      setVisionFollows(key !== currentKey && defaultVisionFollows(key));
    }

    function resetForm() {
      setEdited(false);
    }

    const target = findModel(models, model);
    const showVisionChoice =
      !!settings.vision_enabled && model && model !== currentKey && settings.vision && settings.vision !== model;
    const unchanged =
      model === currentKey &&
      String(budget) === String(settings.chat_budget) &&
      (!native || String(context) === String(loadedContext || ""));

    async function apply() {
      setFormError("");
      var request = {
        model: model,
        context_length: native ? context : null,
        chat_budget: budget,
        vision_follows: showVisionChoice && visionFollows,
      };
      var plan;
      try {
        plan = (await D.postJson("/api/lm-studio/switch-model", Object.assign({ dry_run: true }, request))).plan;
      } catch (err) {
        setFormError(err.message);
        return;
      }
      if (plan.noop) {
        notify("Nothing to change.");
        setEdited(false);
        return;
      }
      var switching = !!(plan.env && plan.env.LOCAL_CHAT);
      var ok = await confirm({
        title: switching ? "Switch the chat model?" : plan.load ? "Reload the chat model?" : "Save the chat prompt budget?",
        message: switching ? "Soupy will write replies with " + plan.model + "." : null,
        details: planLines(plan),
        confirmLabel: switching ? "Switch" : plan.load ? "Reload" : "Save",
      });
      if (!ok) return;
      onBusy(true);
      setWorking(
        plan.load
          ? "LM Studio is loading " + plan.load.key + ". A large model takes a minute or more."
          : "Saving and restarting the bot."
      );
      try {
        var result = await D.postJson("/api/lm-studio/switch-model", request);
        notify(result.message || "Done.");
        setEdited(false);
      } catch (err) {
        notify(err.message, true);
      } finally {
        setWorking("");
        onBusy(false);
        onReload();
        onChanged();
      }
    }

    // ---- What is so right now ----
    var chatFacts;
    var chatNote = null;
    if (!data) {
      chatFacts = null;
    } else if (!settings.chat) {
      chatFacts = e("span", { className: "muted" }, "Not set (LOCAL_CHAT)");
    } else if (!data.reachable) {
      chatFacts = e("span", { className: "muted" }, "Can't tell: the model server is unreachable.");
    } else if (!native) {
      chatFacts = e("span", { className: "muted" }, "This server doesn't report what is loaded.");
    } else if (!current) {
      chatFacts = null;
      chatNote = e(Note, { kind: "error" }, "LM Studio has no model with this name. Pick one below.");
    } else if (!current.loaded) {
      chatFacts = e("span", { className: "muted" }, [current.quant, gb(current.size_bytes)].filter(Boolean).join(" · "));
      chatNote = e(Note, { kind: "warn" }, "Not loaded in LM Studio right now. Apply below to load it.");
    } else {
      chatFacts = e(
        "span",
        { className: "muted" },
        ["Loaded with a " + fmtNum(current.loaded_context) + "-token context", current.quant, gb(current.size_bytes)]
          .filter(Boolean)
          .join(" · ")
      );
      if (current.instances.length > 1) {
        chatNote = e(
          Note,
          { kind: "warn" },
          "LM Studio has " + current.instances.length + " copies of it loaded. Apply below to go back to one."
        );
      }
    }

    var budgetNote = null;
    if (loadedContext && settings.chat_budget > loadedContext) {
      budgetNote = e(
        Note,
        { kind: "error" },
        "The budget is larger than the loaded context (" + tokens(loadedContext) + "). A long conversation overflows the model."
      );
    } else if (loadedContext && settings.chat_budget > loadedContext * HEADROOM) {
      budgetNote = e(
        Note,
        { kind: "warn" },
        "The budget is " + percent(settings.chat_budget, loadedContext) + " of the loaded context. Soupy sizes prompts from character counts, so leave some room."
      );
    }

    var staleBot =
      botRunning && botModel && settings.chat && botModel !== settings.chat
        ? e(
            Note,
            { kind: "warn" },
            "The running bot started with " + botModel + ". It uses " + settings.chat + " after a restart."
          )
        : null;

    var contextMax = target && target.max_context ? target.max_context : null;

    return e(
      "article",
      { className: "model-card" },
      e(
        "div",
        { className: "model-card-head" },
        e("h4", null, "Chat model"),
        e(
          "button",
          { type: "button", className: "secondary", disabled: loading || !!working, onClick: onReload },
          loading ? "Checking…" : "Refresh"
        )
      ),
      !data
        ? D.PanelState({ kind: "loading", message: "Asking the model server…" })
        : data.failed
          ? D.PanelState({ kind: "error", message: data.failed })
          : e(
            React.Fragment,
            null,
            !data.reachable ? D.PanelState({ kind: "error", message: data.error || "The model server can't be reached." }) : null,
            e(
              "dl",
              { className: "model-facts" },
              e("dt", null, "In use"),
              e("dd", null, e("span", { className: "mono model-name" }, settings.chat || "—"), " ", chatFacts),
              e("dt", null, "Chat prompt budget"),
              e(
                "dd",
                null,
                tokens(settings.chat_budget),
                loadedContext
                  ? e("span", { className: "muted" }, " · " + percent(settings.chat_budget, loadedContext) + " of the loaded context")
                  : null
              )
            ),
            chatNote,
            budgetNote,
            staleBot,
            e("h5", { className: "model-subhead" }, "Change it"),
            e(
              "div",
              { className: "model-form" },
              e(
                "div",
                { className: "dash-field model-form-wide" },
                e("label", { htmlFor: "model-select" }, "Model"),
                e(
                  "select",
                  {
                    id: "model-select",
                    value: model,
                    disabled: !!working || !choices.length,
                    onChange: function (ev) {
                      pickModel(ev.target.value);
                    },
                  },
                  choices.length
                    ? choices.map(function (c) {
                        return e("option", { key: c.value, value: c.value }, c.label);
                      })
                    : e("option", { value: "" }, "No models to pick from")
                )
              ),
              native
                ? e(
                    "div",
                    { className: "dash-field" },
                    e("label", { htmlFor: "model-context" }, "Context length to load"),
                    e("input", {
                      id: "model-context",
                      type: "number",
                      min: 512,
                      max: contextMax || undefined,
                      step: 1024,
                      value: context,
                      disabled: !!working,
                      onChange: function (ev) {
                        setEdited(true);
                        setFormError("");
                        setContext(ev.target.value);
                      },
                    }),
                    e(
                      "small",
                      { className: "muted" },
                      contextMax ? "Tokens. This model takes up to " + fmtNum(contextMax) + "." : "Tokens."
                    )
                  )
                : null,
              e(
                "div",
                { className: "dash-field" },
                e("label", { htmlFor: "model-budget" }, "Chat prompt budget"),
                e("input", {
                  id: "model-budget",
                  type: "number",
                  min: 512,
                  step: 500,
                  value: budget,
                  disabled: !!working,
                  onChange: function (ev) {
                    setEdited(true);
                    setFormError("");
                    setBudget(ev.target.value);
                  },
                }),
                e("small", { className: "muted" }, "Tokens. Soupy trims history and search results to fit each reply's prompt under this.")
              ),
              showVisionChoice
                ? e(
                    "label",
                    { className: "model-check model-form-wide" },
                    e("input", {
                      type: "checkbox",
                      checked: visionFollows,
                      disabled: !!working,
                      onChange: function (ev) {
                        setEdited(true);
                        setVisionFollows(ev.target.checked);
                      },
                    }),
                    e(
                      "span",
                      null,
                      "Read images with it too",
                      e(
                        "small",
                        { className: "muted" },
                        visionFollows
                          ? " " + settings.vision + " is unloaded."
                          : " " + settings.vision + " stays loaded for images."
                      )
                    )
                  )
                : null
            ),
            formError ? D.PanelState({ kind: "error", message: formError, inline: true }) : null,
            working ? D.PanelState({ kind: "loading", message: working, inline: true }) : null,
            e(
              "div",
              { className: "dash-controls" },
              e(
                "button",
                { type: "button", disabled: busy || !!working || unchanged || !model, onClick: apply },
                model !== currentKey ? "Switch model…" : "Apply…"
              ),
              edited
                ? e("button", { type: "button", className: "secondary", disabled: !!working, onClick: resetForm }, "Reset")
                : null
            )
          )
    );
  }

  // ---- What the server has loaded -------------------------------------------

  function ServerCard({ data, notify, onReload }) {
    const [saving, setSaving] = useState(false);
    if (!data || data.failed) return e("article", { className: "model-card" }, e("h4", null, "On the model server"));
    const settings = data.settings || {};
    const models = data.models || [];
    const native = !!data.native;

    var rows = [];
    models.forEach(function (m) {
      (m.instances || []).forEach(function (inst) {
        rows.push({ id: inst.id, context: inst.context_length, size: m.size_bytes, roles: m.roles, type: m.type });
      });
    });
    var total = rows.reduce(function (sum, r) {
      return sum + (r.size || 0);
    }, 0);

    function roleState(name, offText) {
      if (!name) return e("span", { className: "muted" }, offText);
      var m = findModel(models, name);
      var state = !data.reachable || !native ? "" : !m ? "not on the server" : m.loaded ? "loaded" : "not loaded";
      return e(
        React.Fragment,
        null,
        e("span", { className: "mono model-name" }, name),
        state ? e("span", { className: state === "loaded" ? "muted" : "model-flag" }, " · " + state) : null
      );
    }

    var chatKeys = models
      .filter(function (m) {
        return m.type !== "embedding";
      })
      .map(function (m) {
        return m.key;
      });
    var saved = settings.saved_list || [];
    var listStale =
      native &&
      chatKeys.length > 0 &&
      (saved.length !== chatKeys.length ||
        chatKeys.some(function (k) {
          return saved.indexOf(k) === -1;
        }));

    async function updateSavedList() {
      setSaving(true);
      try {
        await D.postJson("/api/env/save", { AVAILABLE_MODELS: chatKeys.join(", ") });
        notify("Saved " + D.plural(chatKeys.length, "model name") + " to AVAILABLE_MODELS.");
        onReload();
      } catch (err) {
        notify("Could not save the list: " + err.message, true);
      } finally {
        setSaving(false);
      }
    }

    var embeddingCount = models.length - chatKeys.length;

    return e(
      "article",
      { className: "model-card" },
      e("div", { className: "model-card-head" }, e("h4", null, "On the model server")),
      e(
        "dl",
        { className: "model-facts" },
        e("dt", null, "Images"),
        e(
          "dd",
          null,
          settings.vision_enabled
            ? settings.vision === settings.chat
              ? e("span", null, "The chat model")
              : roleState(settings.vision, "Not set")
            : e("span", { className: "muted" }, "Off (ENABLE_VISION)")
        ),
        e("dt", null, "Embeddings"),
        e("dd", null, roleState(settings.embedding, "Not set (RAG_EMBEDDING_MODEL)"))
      ),
      !data.reachable
        ? null
        : !native
          ? e(
              Note,
              null,
              "This server lists " + D.plural(models.length, "model") + " but doesn't report what is loaded or load one on request."
            )
          : e(
              React.Fragment,
              null,
              e("h5", { className: "model-subhead" }, "Loaded now"),
              rows.length
                ? e(
                    "table",
                    { className: "model-loaded" },
                    e(
                      "tbody",
                      null,
                      rows.map(function (r) {
                        return e(
                          "tr",
                          { key: r.id },
                          e("td", { className: "mono model-name" }, r.id),
                          e("td", null, e(RoleChips, { roles: r.roles })),
                          e("td", { className: "model-num" }, r.context ? tokens(r.context) : ""),
                          e("td", { className: "model-num" }, gb(r.size))
                        );
                      })
                    )
                  )
                : e(Note, { kind: "warn" }, "Nothing is loaded."),
              e(
                "p",
                { className: "muted model-foot" },
                (rows.length > 1 && total ? gb(total) + " of models loaded. " : "") +
                  D.plural(chatKeys.length, "chat model") +
                  " and " +
                  D.plural(embeddingCount, "embedding model") +
                  " downloaded."
              ),
              listStale
                ? e(
                    "p",
                    { className: "muted model-foot" },
                    "The saved list for when the server is unreachable (AVAILABLE_MODELS) has " +
                      D.plural(saved.length, "name") +
                      " and doesn't match. ",
                    e(
                      "button",
                      { type: "button", className: "dash-plain model-link", disabled: saving, onClick: updateSavedList },
                      saving ? "Saving…" : "Update it"
                    )
                  )
                : null
            ),
      e(
        "p",
        { className: "muted model-foot" },
        "Temperature, reply length and the image and embedding models are in ",
        e("a", { href: "/env" }, "Configuration"),
        "."
      )
    );
  }

  // ---- Prompts ---------------------------------------------------------------

  const GROUPS = [
    { key: "personality", label: "Personality" },
    { key: "images", label: "Image generation" },
  ];

  function PromptEditor({ chatBudget, botRunning, botStartedAt, busy, onBusy, notify, confirm, onStatus }) {
    const [prompts, setPrompts] = useState(null);
    const [loadError, setLoadError] = useState("");
    const [selected, setSelected] = useState("behaviour");
    // Unsaved text per prompt, so switching between prompts loses nothing.
    const [drafts, setDrafts] = useState({});
    const [working, setWorking] = useState(false);
    const editorRef = useRef(null);

    const load = useCallback(async function () {
      try {
        const data = await D.readJson("/api/prompts/get");
        if (!Array.isArray(data.prompts)) throw new Error(OLD_SERVER);
        setPrompts(data.prompts);
        setLoadError("");
      } catch (err) {
        setLoadError(err.message);
      }
    }, []);

    useEffect(
      function () {
        load();
      },
      [load]
    );

    const current =
      (prompts || []).find(function (p) {
        return p.name === selected;
      }) || null;
    const stored = current ? current.value : "";
    const hasDraft = Object.prototype.hasOwnProperty.call(drafts, selected);
    const text = hasDraft ? drafts[selected] : stored;
    const dirty = hasDraft && drafts[selected] !== stored;
    const overridden = !!current && current.source === "env";

    function isDirty(p) {
      return Object.prototype.hasOwnProperty.call(drafts, p.name) && drafts[p.name] !== p.value;
    }
    const anyDirty = (prompts || []).some(isDirty);

    // The browser asks before a reload or a closed tab throws edits away.
    useEffect(
      function () {
        if (!anyDirty) return;
        function warn(ev) {
          ev.preventDefault();
          ev.returnValue = "";
        }
        window.addEventListener("beforeunload", warn);
        return function () {
          window.removeEventListener("beforeunload", warn);
        };
      },
      [anyDirty]
    );

    function setDraft(name, value) {
      setDrafts(function (old) {
        var next = Object.assign({}, old);
        if (value === null) delete next[name];
        else next[name] = value;
        return next;
      });
    }

    function replacePrompt(fresh) {
      setPrompts(function (old) {
        return (old || []).map(function (p) {
          return p.name === fresh.name ? fresh : p;
        });
      });
    }

    async function save(restart) {
      if (!current || !dirty || working) return;
      if (!text.trim()) {
        var sure = await confirm({
          title: "Save an empty prompt?",
          message:
            "Soupy would run with no text for " +
            current.label +
            '. To go back to the text that ships with Soupy, use "Use shipped default" instead.',
          confirmLabel: "Save it empty",
          danger: true,
        });
        if (!sure) return;
      }
      setWorking(true);
      onBusy(true);
      try {
        const data = await D.postJson("/api/prompts/save", { name: current.name, value: text });
        replacePrompt(data.prompt);
        setDraft(current.name, null);
        var message = "Saved " + data.prompt.file + ".";
        if (data.backup) message += " The version it replaced is in prompts/backups/.";
        if (restart) {
          try {
            onStatus(await restartBot());
            message += " The bot restarted.";
          } catch (err) {
            notify(message + " The bot did not restart: " + err.message, true);
            return;
          }
        } else if (botRunning) {
          message += " The bot uses it after its next restart.";
        }
        notify(message);
      } catch (err) {
        notify("Save failed: " + err.message, true);
      } finally {
        setWorking(false);
        onBusy(false);
      }
    }

    async function useDefault() {
      if (!current || working) return;
      var ok = await confirm({
        title: "Use the shipped default?",
        message:
          current.file +
          " moves to prompts/backups/ and Soupy goes back to the text that ships with it for " +
          current.label +
          "." +
          (dirty ? " Your unsaved edits here are dropped." : ""),
        confirmLabel: "Use the default",
        danger: true,
      });
      if (!ok) return;
      setWorking(true);
      try {
        const data = await D.postJson("/api/prompts/reset", { name: current.name });
        replacePrompt(data.prompt);
        setDraft(current.name, null);
        notify(
          "Using the shipped default for " +
            current.label +
            ". Your version is in prompts/backups/." +
            (botRunning ? " The bot uses it after its next restart." : "")
        );
      } catch (err) {
        notify("Could not switch to the default: " + err.message, true);
      } finally {
        setWorking(false);
      }
    }

    async function restartNow() {
      setWorking(true);
      onBusy(true);
      try {
        onStatus(await restartBot());
        notify("The bot restarted.");
      } catch (err) {
        notify("Restart failed: " + err.message, true);
      } finally {
        setWorking(false);
        onBusy(false);
      }
    }

    if (loadError && !prompts) {
      return e(
        "article",
        { className: "personality-panel" },
        e("h4", null, "Prompts"),
        D.PanelState({ kind: "error", message: loadError })
      );
    }
    if (!prompts) {
      return e(
        "article",
        { className: "personality-panel" },
        e("h4", null, "Prompts"),
        D.PanelState({ kind: "loading", message: "Loading prompts…" })
      );
    }

    var source = null;
    if (current) {
      if (overridden) {
        source = e(
          Note,
          { kind: "warn" },
          "This text comes from ",
          e("span", { className: "mono" }, current.env_var),
          " in .env-stable, which overrides ",
          e("span", { className: "mono" }, current.file),
          ". Remove that line to edit the prompt here (",
          e("span", { className: "mono" }, "python tools/migrate_prompts.py"),
          " moves it into the file for you)."
        );
      } else if (current.source === "custom") {
        source = e(
          "p",
          { className: "muted prompt-source" },
          "Your version, in ",
          e("span", { className: "mono" }, current.file),
          current.modified_at ? ". Saved " + D.timeAgo(current.modified_at) + "." : "."
        );
      } else if (current.source === "default") {
        source = e(
          "p",
          { className: "muted prompt-source" },
          "The text that ships with Soupy. Saving a change creates ",
          e("span", { className: "mono" }, current.file),
          "."
        );
      } else {
        source = e(
          "p",
          { className: "muted prompt-source" },
          "No text yet. Saving creates ",
          e("span", { className: "mono" }, current.file),
          "."
        );
      }
    }

    // The bot reads prompts once, when it starts.
    var savedAt = current && current.modified_at ? D.parseUtc(current.modified_at) : null;
    var startedAt = botStartedAt ? D.parseUtc(botStartedAt) : null;
    var stale = botRunning && !overridden && savedAt && startedAt && savedAt > startedAt;

    var approxTokens = Math.max(1, Math.round(text.length / CHARS_PER_TOKEN));
    var counts =
      fmtNum(text.length) +
      " characters · " +
      D.plural(text ? text.replace(/\n$/, "").split("\n").length : 0, "line") +
      " · about " +
      tokens(text ? approxTokens : 0);
    if (current && current.name === "behaviour" && chatBudget && text) {
      counts += ", " + percent(approxTokens, chatBudget) + " of the chat prompt budget";
    }

    var canReset = !!current && current.has_custom && current.default !== null && !overridden;
    var showDefault = canReset && current.default !== stored;

    return e(
      "article",
      { className: "personality-panel" },
      e("h4", null, "Prompts"),
      e(
        "p",
        { className: "muted prompt-intro" },
        "The text Soupy hands the model. The bot reads these files when it starts, so a saved change takes effect at the next restart."
      ),
      e(
        "div",
        { className: "personality-toolbar dash-controls" },
        e(
          "div",
          { className: "dash-field prompt-pick" },
          e("label", { htmlFor: "personality-select" }, "Prompt"),
          e(
            "select",
            {
              id: "personality-select",
              value: selected,
              onChange: function (ev) {
                setSelected(ev.target.value);
              },
            },
            GROUPS.map(function (group) {
              return e(
                "optgroup",
                { key: group.key, label: group.label },
                prompts
                  .filter(function (p) {
                    return p.group === group.key;
                  })
                  .map(function (p) {
                    return e("option", { key: p.name, value: p.name }, p.label + (isDirty(p) ? " (unsaved)" : ""));
                  })
              );
            })
          )
        ),
        e(
          "button",
          {
            type: "button",
            className: "secondary",
            disabled: working,
            title: "Read the files again. Unsaved edits stay in the editor.",
            onClick: function () {
              load().then(function () {
                notify("Read the prompt files again.");
              });
            },
          },
          "Reload from disk"
        )
      ),
      current ? e("p", { className: "prompt-used" }, e("strong", null, "Used for: "), current.used_for) : null,
      source,
      stale
        ? e(
            Note,
            { kind: "warn" },
            "Saved after the bot started, so the bot is still using the earlier text. ",
            e(
              "button",
              { type: "button", className: "dash-plain model-link", disabled: busy || working, onClick: restartNow },
              "Restart the bot"
            )
          )
        : null,
      e(
        "div",
        { className: "dash-field" },
        e("label", { htmlFor: "personality-body", className: "prompt-label" }, current ? current.label : "Prompt"),
        e("textarea", {
          id: "personality-body",
          ref: editorRef,
          className: "personality-editor",
          value: text,
          readOnly: overridden,
          spellCheck: false,
          onChange: function (ev) {
            var value = ev.target.value.replace(/\r\n?/g, "\n");
            setDraft(selected, value === stored ? null : value);
          },
          onKeyDown: function (ev) {
            if ((ev.metaKey || ev.ctrlKey) && String(ev.key).toLowerCase() === "s") {
              ev.preventDefault();
              save(false);
            }
          },
        })
      ),
      e(
        "div",
        { className: "prompt-status" },
        e("span", { className: "muted" }, counts),
        dirty ? e("span", { className: "prompt-unsaved" }, "Unsaved changes") : null
      ),
      e(
        "div",
        { className: "dash-controls prompt-actions" },
        e(
          "button",
          {
            type: "button",
            className: "secondary",
            disabled: !dirty || working,
            onClick: function () {
              setDraft(selected, null);
            },
          },
          "Discard changes"
        ),
        e(
          "button",
          {
            type: "button",
            className: "secondary",
            disabled: !dirty || working || overridden,
            onClick: function () {
              save(false);
            },
          },
          working ? "Saving…" : "Save"
        ),
        e(
          "button",
          {
            type: "button",
            disabled: !dirty || working || busy || overridden,
            onClick: function () {
              save(true);
            },
          },
          "Save + restart bot"
        ),
        e("span", { className: "prompt-actions-gap" }),
        canReset
          ? e(
              "button",
              { type: "button", className: "dash-btn-danger-outline", disabled: working, onClick: useDefault },
              "Use shipped default…"
            )
          : null
      ),
      showDefault
        ? e(
            "details",
            { className: "prompt-default" },
            e("summary", null, "The shipped default, for comparison"),
            e("pre", null, current.default)
          )
        : null
    );
  }

  // ---- The tab ---------------------------------------------------------------

  /** Stays mounted while another tab is showing, so unsaved prompt edits survive a look elsewhere. */
  function ModelTab(props) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const ticket = useRef(0);

    const reload = useCallback(async function () {
      const mine = ++ticket.current;
      setLoading(true);
      try {
        const fresh = await D.readJson("/api/lm-studio/models");
        if (!fresh.settings) throw new Error(OLD_SERVER);
        if (mine === ticket.current) setData(fresh);
      } catch (err) {
        // Nothing to show but the reason: the request itself failed.
        if (mine === ticket.current) setData({ failed: err.message });
      } finally {
        if (mine === ticket.current) setLoading(false);
      }
    }, []);

    useEffect(
      function () {
        if (props.active) reload();
      },
      [props.active, reload]
    );

    return e(
      "section",
      // Hidden with a style: the stylesheet's display rule would beat the hidden attribute.
      { className: "dash-app", style: props.active ? null : { display: "none" } },
      e(
        "div",
        { className: "model-grid" },
        e(ModelCard, {
          data: data,
          loading: loading,
          onReload: reload,
          botRunning: props.botRunning,
          botModel: props.botModel,
          busy: props.busy,
          onBusy: props.onBusy,
          notify: props.notify,
          confirm: props.confirm,
          onChanged: props.onRefreshStatus,
        }),
        e(ServerCard, { data: data, notify: props.notify, onReload: reload })
      ),
      e(PromptEditor, {
        chatBudget: data && data.settings ? data.settings.chat_budget : null,
        botRunning: props.botRunning,
        botStartedAt: props.botStartedAt,
        busy: props.busy,
        onBusy: props.onBusy,
        notify: props.notify,
        confirm: props.confirm,
        onStatus: props.onStatus,
      })
    );
  }

  D.ModelTab = ModelTab;
})();
