/**
 * IPMA — Industrial Plant Monitoring Agent
 * Frontend Client Application
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chatForm");
  const queryInput = document.getElementById("queryInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeCard = document.getElementById("welcomeCard");
  const queryProgressBar = document.getElementById("queryProgressBar");
  const progressText = document.getElementById("progressText");
  const resetSessionBtn = document.getElementById("resetSessionBtn");
  const refreshFleetBtn = document.getElementById("refreshFleetBtn");
  const hxAssetCards = document.getElementById("hxAssetCards");
  const bearingAssetCards = document.getElementById("bearingAssetCards");
  const telemetryTimestamp = document.getElementById("telemetryTimestamp");
  const activeModelTag = document.getElementById("activeModelTag");

  // Configure Marked for Markdown rendering
  if (typeof marked !== "undefined") {
    marked.setOptions({
      gfm: true,
      breaks: true,
      sanitize: false,
    });
  }

  // Auto-resize textarea
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
  });

  // Keyboard shortcut: Enter to submit (Shift+Enter for newline)
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // -------------------------------------------------------------------------
  // Telemetry Fleet Data Loader
  // -------------------------------------------------------------------------
  async function loadFleetData() {
    telemetryTimestamp.textContent = "REFRESHING...";
    try {
      const res = await fetch("/api/fleet");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderFleetAssets(data);
      telemetryTimestamp.textContent = "LIVE · " + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (err) {
      console.warn("Fleet data load error:", err);
      telemetryTimestamp.textContent = "OFFLINE";
      renderFleetFallback();
    }
  }

  function renderFleetAssets(data) {
    const snap = data.snapshot || {};
    const hxSnap = snap.heat_exchangers || [];
    const brgSnap = snap.bearing_tests || {};

    // Render Heat Exchangers
    if (hxSnap.length > 0) {
      hxAssetCards.innerHTML = hxSnap.map(hx => {
        const id = hx.exchanger_id;
        const deltaT = (hx.delta_T_tube != null) ? hx.delta_T_tube.toFixed(1) : "—";
        const flow = (hx.mass_flow != null) ? hx.mass_flow.toFixed(0) : "—";
        
        // Severity heuristic for UI tag
        let tagClass = "tag-clean";
        let tagLabel = "NORMAL";
        if (id === "E02") {
          tagClass = "tag-warning";
          tagLabel = "FOULING";
        } else if (id === "E05") {
          tagClass = "tag-warning";
          tagLabel = "ALERT";
        }

        return `
          <div class="asset-card" data-asset-type="hx" data-asset-id="${id}">
            <div class="asset-header">
              <span class="asset-id">${id}</span>
              <span class="asset-tag ${tagClass}">${tagLabel}</span>
            </div>
            <div class="asset-metrics">
              <span>ΔT: ${deltaT}°C</span>
              <span>Flow: ${flow} kg/s</span>
            </div>
          </div>
        `;
      }).join("");
    }

    // Render Bearing Tests
    const bearingItems = [
      { id: "Test 1", testNum: 1, bearings: "4 Bearings (8-ch)", status: "COMPLETED", tagClass: "tag-zone-a" },
      { id: "Test 2", testNum: 2, bearings: "Bearing 1 Failed", status: "OUTER RACE", tagClass: "tag-danger" },
      { id: "Test 3", testNum: 3, bearings: "Bearing 3 Flaw", status: "ZONE C", tagClass: "tag-warning" },
    ];

    bearingAssetCards.innerHTML = bearingItems.map(item => {
      return `
        <div class="asset-card" data-asset-type="bearing" data-test-id="${item.testNum}">
          <div class="asset-header">
            <span class="asset-id">${item.id}</span>
            <span class="asset-tag ${item.tagClass}">${item.status}</span>
          </div>
          <div class="asset-metrics">
            <span>${item.bearings}</span>
          </div>
        </div>
      `;
    }).join("");

    // Attach click listeners to cards to populate chat
    document.querySelectorAll(".asset-card").forEach(card => {
      card.addEventListener("click", () => {
        const hxId = card.getAttribute("data-asset-id");
        const testId = card.getAttribute("data-test-id");
        if (hxId) {
          submitQuery(`Perform a comprehensive fouling and thermal efficiency analysis for Heat Exchanger ${hxId}. What is its energy penalty and cleaning recommendation?`);
        } else if (testId) {
          submitQuery(`Analyze Bearing Test ${testId}. Which bearing is at highest risk, what is its ISO 10816 severity zone, and what is the estimated RUL?`);
        }
      });
    });
  }

  function renderFleetFallback() {
    hxAssetCards.innerHTML = `
      <div class="asset-card" data-asset-id="E02">
        <div class="asset-header"><span class="asset-id">E01 - E05</span><span class="asset-tag tag-clean">READY</span></div>
        <div class="asset-metrics"><span>5 Exchangers</span><span>Campaign Data</span></div>
      </div>
    `;
    bearingAssetCards.innerHTML = `
      <div class="asset-card" data-test-id="2">
        <div class="asset-header"><span class="asset-id">Test 1 - 3</span><span class="asset-tag tag-warning">READY</span></div>
        <div class="asset-metrics"><span>NASA Run-to-Failure</span></div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // Chat Message Rendering
  // -------------------------------------------------------------------------
  function appendMessage(sender, text, timestamp) {
    if (welcomeCard && welcomeCard.style.display !== "none") {
      welcomeCard.style.display = "none";
    }

    const row = document.createElement("div");
    row.className = `message-row ${sender}-row`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const authorName = sender === "user" ? "OPERATOR" : "IPMA AGENT";
    const timeStr = timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    meta.innerHTML = `<span class="meta-author">${authorName}</span><span>·</span><span>${timeStr}</span>`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (sender === "user") {
      bubble.textContent = text;
    } else {
      // Render Markdown for Agent responses
      if (typeof marked !== "undefined") {
        bubble.innerHTML = marked.parse(text);
      } else {
        bubble.textContent = text;
      }

      // Add Copy Button
      const actions = document.createElement("div");
      actions.className = "bubble-actions";
      const copyBtn = document.createElement("button");
      copyBtn.className = "btn-copy";
      copyBtn.textContent = "Copy Report";
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(text).then(() => {
          copyBtn.textContent = "Copied!";
          setTimeout(() => { copyBtn.textContent = "Copy Report"; }, 2000);
        });
      });
      actions.appendChild(copyBtn);
      bubble.appendChild(actions);
    }

    row.appendChild(meta);
    row.appendChild(bubble);
    messagesContainer.appendChild(row);

    // Smooth scroll to bottom
    messagesContainer.scrollTo({
      top: messagesContainer.scrollHeight,
      behavior: "smooth"
    });
  }

  // -------------------------------------------------------------------------
  // Form Submission & Query Pipeline
  // -------------------------------------------------------------------------
  async function submitQuery(messageText) {
    const text = messageText.trim();
    if (!text) return;

    // Display user message immediately
    appendMessage("user", text);

    // Clear input
    queryInput.value = "";
    queryInput.style.height = "auto";
    sendBtn.disabled = true;

    // Temporarily disable quick runbook buttons to avoid rapid burst rate limits
    const actionBtns = document.querySelectorAll(".runbook-btn, .asset-card, .cap-card");
    actionBtns.forEach(b => { b.style.pointerEvents = "none"; b.style.opacity = "0.7"; });

    // Show telemetry progress
    queryProgressBar.style.display = "flex";
    progressText.textContent = "Executing engineering tools & querying plant telemetry...";

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      appendMessage("agent", data.response);
      if (data.model) {
        activeModelTag.textContent = data.model.toUpperCase();
      }
    } catch (err) {
      appendMessage("agent", `⚠️ **Diagnostic Notice**\n\n${err.message}`);
    } finally {
      queryProgressBar.style.display = "none";
      sendBtn.disabled = false;
      actionBtns.forEach(b => { b.style.pointerEvents = "auto"; b.style.opacity = "1"; });
      queryInput.focus();
    }
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitQuery(queryInput.value);
  });

  // -------------------------------------------------------------------------
  // Quick Runbook & Capability Card Clicks
  // -------------------------------------------------------------------------
  document.querySelectorAll(".runbook-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const query = btn.getAttribute("data-query");
      if (query) submitQuery(query);
    });
  });

  document.querySelectorAll(".cap-card").forEach(card => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      if (prompt) submitQuery(prompt);
    });
  });

  // -------------------------------------------------------------------------
  // Session Reset
  // -------------------------------------------------------------------------
  resetSessionBtn.addEventListener("click", async () => {
    if (confirm("Reset conversation and clear diagnostic context?")) {
      try {
        await fetch("/api/reset", { method: "POST" });
        messagesContainer.innerHTML = "";
        if (welcomeCard) {
          welcomeCard.style.display = "flex";
          messagesContainer.appendChild(welcomeCard);
        }
      } catch (err) {
        console.error("Reset failed:", err);
      }
    }
  });

  // Refresh Fleet Button
  refreshFleetBtn.addEventListener("click", () => {
    loadFleetData();
  });

  // Initial Load
  loadFleetData();
});
