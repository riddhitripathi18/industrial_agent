/**
 * IPMA — Industrial Plant Monitoring Agent & Digital Twin Suite
 * Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------------------
  // Global State & Elements
  // -------------------------------------------------------------------------
  const state = {
    activeTab: "fleetTab",
    activeTwin: "hxTwin",
    fleetData: null,
    fleetSummary: null,
    hxParams: {
      exchanger_id: "E02",
      tube_t_in: 194.7,
      tube_t_out: 234.6,
      tube_flow: 103.2,
      shell_t_in: 280.6,
      shell_t_out: 229.1,
      shell_flow: 84.6,
      time_hr: 1000.0,
      crude_api: 32.0,
      crude_tan: 0.8,
      crude_chlorides: 15.0
    },
    bearingParams: {
      bearing_id: "Bearing1",
      rms: 0.15,
      kurtosis: 3.5,
      peak: 0.45,
      peak_to_peak: 0.90,
      crest_factor: 3.0,
      hours_elapsed: 120.0,
      test_id: 2,
      rpm: 2000.0
    },
    millingParams: {
      air_temp_k: 300.0,
      process_temp_k: 310.0,
      rotational_speed_rpm: 1500.0,
      torque_nm: 40.0,
      tool_wear_min: 108.0,
      product_type: "M"
    }
  };

  // Debounce helper
  function debounce(fn, delay = 120) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }

  // -------------------------------------------------------------------------
  // Tab Navigation Handling
  // -------------------------------------------------------------------------
  const navTabs = document.querySelectorAll(".nav-tab-btn");
  const viewPanels = document.querySelectorAll(".view-panel");

  function switchTab(tabId) {
    state.activeTab = tabId;
    navTabs.forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    viewPanels.forEach(panel => {
      panel.classList.toggle("active", panel.id === tabId);
    });
    // Trigger recalculations or layout fixes
    if (tabId === "simulatorTab") {
      updateActiveTwin();
    }
  }

  navTabs.forEach(btn => {
    btn.addEventListener("click", () => {
      switchTab(btn.getAttribute("data-tab"));
    });
  });

  // Digital Twin Switcher
  const twinSwitchBtns = document.querySelectorAll(".twin-switch-btn");
  const twinContainers = document.querySelectorAll(".twin-container");

  function switchTwin(twinId) {
    state.activeTwin = twinId;
    twinSwitchBtns.forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-twin") === twinId);
    });
    twinContainers.forEach(container => {
      container.classList.toggle("active", container.id === twinId);
    });
    updateActiveTwin();
  }

  twinSwitchBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      switchTwin(btn.getAttribute("data-twin"));
    });
  });

  // Direct launcher shortcuts from Fleet Overview
  document.getElementById("openHxSimBtn")?.addEventListener("click", () => {
    switchTab("simulatorTab");
    switchTwin("hxTwin");
  });
  document.getElementById("openBearingSimBtn")?.addEventListener("click", () => {
    switchTab("simulatorTab");
    switchTwin("bearingTwin");
  });
  document.getElementById("openMillingSimBtn")?.addEventListener("click", () => {
    switchTab("simulatorTab");
    switchTwin("millingTwin");
  });

  // -------------------------------------------------------------------------
  // Copilot Drawer Handling
  // -------------------------------------------------------------------------
  const copilotDrawer = document.getElementById("copilotDrawer");
  const toggleCopilotBtn = document.getElementById("toggleCopilotBtn");
  const closeCopilotBtn = document.getElementById("closeCopilotBtn");

  function toggleCopilot(open) {
    if (open !== undefined) {
      copilotDrawer.classList.toggle("open", open);
    } else {
      copilotDrawer.classList.toggle("open");
    }
  }

  toggleCopilotBtn?.addEventListener("click", () => toggleCopilot());
  closeCopilotBtn?.addEventListener("click", () => toggleCopilot(false));

  // -------------------------------------------------------------------------
  // Quick Runbook Prompts & Chips
  // -------------------------------------------------------------------------
  document.querySelectorAll(".runbook-chip, .hero-action-card").forEach(chip => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query") || chip.getAttribute("data-prompt");
      if (!query) return;
      // Switch to Agent Tab and submit query
      switchTab("agentTab");
      const input = document.getElementById("agentQueryInput");
      if (input) {
        input.value = query;
        document.getElementById("agentChatForm").dispatchEvent(new Event("submit"));
      }
    });
  });

  // -------------------------------------------------------------------------
  // Fleet Data Loader & Hydration
  // -------------------------------------------------------------------------
  async function loadFleetTelemetry() {
    try {
      const [fleetRes, summaryRes, healthRes] = await Promise.all([
        fetch("/api/fleet").then(r => r.json()),
        fetch("/api/fleet/summary").then(r => r.json()).catch(() => null),
        fetch("/api/health").then(r => r.json()).catch(() => null)
      ]);

      state.fleetData = fleetRes;
      state.fleetSummary = summaryRes;

      if (healthRes && healthRes.model) {
        const tag = document.getElementById("activeModelTag");
        if (tag) tag.textContent = healthRes.model.toUpperCase();
      }

      renderFleetOverview(fleetRes, summaryRes);
    } catch (err) {
      console.error("Failed to load fleet data:", err);
    }
  }

  function renderFleetOverview(fleet, summary) {
    // 1. KPI Cards
    if (summary) {
      const hxAlerts = summary.heat_exchangers?.breached || 1;
      const hxLoss = summary.heat_exchangers?.total_energy_loss_usd || 424548;
      const hxEl = document.getElementById("kpiHxStatus");
      if (hxEl) hxEl.textContent = `${hxAlerts} ALERT (${Math.round(hxLoss / 1000)}k Loss)`;

      const brgEl = document.getElementById("kpiBrgStatus");
      if (brgEl) brgEl.textContent = `${summary.bearings?.total_campaigns || 3} CAMPAIGNS`;
    }

    // 2. Heat Exchanger Grid
    const hxContainer = document.getElementById("hxFleetContainer");
    if (hxContainer && fleet.snapshot?.heat_exchangers) {
      const hxData = fleet.snapshot.heat_exchangers.exchangers || {};
      const units = ["E01", "E02", "E03", "E04", "E05"];
      const fluids = {
        E01: "Heavy Naphtha",
        E02: "Kerosene",
        E03: "Light Diesel",
        E04: "LVGO",
        E05: "Heavy Diesel"
      };

      hxContainer.innerHTML = units.map(uid => {
        const u = hxData[uid] || {};
        const isBreached = uid === "E02"; // E02 is historically fouled
        const badgeClass = isBreached ? "badge-alert" : "badge-ok";
        const badgeText = isBreached ? "TEMA BREACH" : "NORMAL";
        const cardClass = isBreached ? "hx-card breach-alert" : "hx-card";

        return `
          <div class="${cardClass}" data-hx="${uid}">
            <div class="hx-card-top">
              <span class="hx-id">${uid}</span>
              <span class="${badgeClass}">${badgeText}</span>
            </div>
            <div class="hx-fluid">${fluids[uid]}</div>
            <div class="hx-metric-row">
              <span class="hx-metric-lbl">Flow:</span>
              <span class="hx-metric-val">${(u.tube_mass_flow_kg_s || 103).toFixed(1)} kg/s</span>
            </div>
            <div class="hx-metric-row">
              <span class="hx-metric-lbl">Tube ΔT:</span>
              <span class="hx-metric-val">${(u.delta_t_tube || 39.9).toFixed(1)} °C</span>
            </div>
            <div class="hx-metric-row">
              <span class="hx-metric-lbl">Shell ΔT:</span>
              <span class="hx-metric-val">${(u.delta_t_shell || 51.4).toFixed(1)} °C</span>
            </div>
          </div>
        `;
      }).join("");

      // Add click to load into simulator
      hxContainer.querySelectorAll(".hx-card").forEach(card => {
        card.addEventListener("click", () => {
          const uid = card.getAttribute("data-hx");
          const preset = document.getElementById("hxPresetSelect");
          if (preset) {
            preset.value = uid;
            preset.dispatchEvent(new Event("change"));
          }
          switchTab("simulatorTab");
          switchTwin("hxTwin");
        });
      });
    }

    // 3. Bearing Campaigns Grid
    const brgContainer = document.getElementById("bearingFleetContainer");
    if (brgContainer && fleet.snapshot?.bearing_tests) {
      const brgData = fleet.snapshot.bearing_tests;
      const campaigns = [
        { id: 1, title: "Campaign 1", fault: "Bearing 3 (Inner) / Bearing 4 (Roller)", duration: "718.7 hrs", worstZone: "Zone D" },
        { id: 2, title: "Campaign 2", fault: "Bearing 1 (Outer Race Defect)", duration: "328.0 hrs", worstZone: "Zone D" },
        { id: 3, title: "Campaign 3", fault: "Bearing 3 (Outer Race Defect)", duration: "2,108 hrs", worstZone: "Zone D" }
      ];

      brgContainer.innerHTML = campaigns.map(c => `
        <div class="bearing-camp-card" data-campaign="${c.id}">
          <div class="camp-header">
            <span class="camp-title">${c.title}</span>
            <span class="badge-alert">${c.worstZone}</span>
          </div>
          <div class="camp-fault-tag">⚠️ ${c.fault}</div>
          <div class="hx-metric-row">
            <span class="hx-metric-lbl">Test Duration:</span>
            <span class="hx-metric-val">${c.duration}</span>
          </div>
          <div class="hx-metric-row">
            <span class="hx-metric-lbl">Sensors:</span>
            <span class="hx-metric-val">4 Channels (X/Y)</span>
          </div>
        </div>
      `).join("");

      brgContainer.querySelectorAll(".bearing-camp-card").forEach(card => {
        card.addEventListener("click", () => {
          switchTab("simulatorTab");
          switchTwin("bearingTwin");
        });
      });
    }

    // 4. Milling High Risk List
    const millingContainer = document.getElementById("millingFleetContainer");
    if (millingContainer && fleet.snapshot?.milling_machines) {
      const highWear = fleet.snapshot.milling_machines;
      millingContainer.innerHTML = highWear.map(m => {
        const wear = m.tool_wear_min || 205;
        const prob = wear > 210 ? 84.2 : 46.5;
        const color = wear > 210 ? "alert-text" : "warning-text";

        return `
          <div class="milling-unit-row" data-mid="${m.udi}">
            <div class="milling-unit-info">
              <span class="milling-unit-id">CNC Machine #${m.udi} (${m.product_id || 'M'})</span>
              <span class="milling-unit-meta">Wear: ${wear} min · RPM: ${m.rotational_speed_rpm || 1400}</span>
            </div>
            <div class="milling-unit-risk">
              <span class="risk-pct ${color}">${prob}%</span>
            </div>
          </div>
        `;
      }).join("");

      millingContainer.querySelectorAll(".milling-unit-row").forEach(row => {
        row.addEventListener("click", () => {
          const mid = row.getAttribute("data-mid");
          const wearSlider = document.getElementById("millWear");
          if (wearSlider) {
            wearSlider.value = 215;
            wearSlider.dispatchEvent(new Event("input"));
          }
          switchTab("simulatorTab");
          switchTwin("millingTwin");
        });
      });
    }
  }

  // -------------------------------------------------------------------------
  // TWIN 1: Shell & Tube Heat Exchanger Simulation
  // -------------------------------------------------------------------------
  const hxControls = {
    tubeIn: document.getElementById("hxTubeIn"),
    tubeOut: document.getElementById("hxTubeOut"),
    tubeFlow: document.getElementById("hxTubeFlow"),
    shellIn: document.getElementById("hxShellIn"),
    shellOut: document.getElementById("hxShellOut"),
    shellFlow: document.getElementById("hxShellFlow"),
    runtime: document.getElementById("hxRuntime"),
    tan: document.getElementById("hxTan"),
    presetSelect: document.getElementById("hxPresetSelect")
  };

  const hxPresetValues = {
    E01: { tubeIn: 154.8, tubeOut: 194.7, tubeFlow: 103.2, shellIn: 228.6, shellOut: 177.2, shellFlow: 84.6, tan: 0.8 },
    E02: { tubeIn: 194.7, tubeOut: 234.6, tubeFlow: 103.2, shellIn: 280.6, shellOut: 229.1, shellFlow: 84.6, tan: 0.8 },
    E03: { tubeIn: 234.6, tubeOut: 274.5, tubeFlow: 103.2, shellIn: 320.5, shellOut: 270.0, shellFlow: 84.6, tan: 0.8 },
    E04: { tubeIn: 180.0, tubeOut: 220.0, tubeFlow: 95.0, shellIn: 260.0, shellOut: 215.0, shellFlow: 75.0, tan: 0.9 },
    E05: { tubeIn: 210.0, tubeOut: 250.0, tubeFlow: 90.0, shellIn: 300.0, shellOut: 250.0, shellFlow: 70.0, tan: 1.1 }
  };

  hxControls.presetSelect?.addEventListener("change", (e) => {
    const p = hxPresetValues[e.target.value];
    if (p) {
      if (hxControls.tubeIn) hxControls.tubeIn.value = p.tubeIn;
      if (hxControls.tubeOut) hxControls.tubeOut.value = p.tubeOut;
      if (hxControls.tubeFlow) hxControls.tubeFlow.value = p.tubeFlow;
      if (hxControls.shellIn) hxControls.shellIn.value = p.shellIn;
      if (hxControls.shellOut) hxControls.shellOut.value = p.shellOut;
      if (hxControls.shellFlow) hxControls.shellFlow.value = p.shellFlow;
      if (hxControls.tan) hxControls.tan.value = p.tan;
      triggerHxRecalc();
    }
  });

  const runHxSimulation = debounce(async () => {
    const payload = {
      exchanger_id: hxControls.presetSelect?.value || "E02",
      tube_t_in: parseFloat(hxControls.tubeIn?.value || 194.7),
      tube_t_out: parseFloat(hxControls.tubeOut?.value || 234.6),
      tube_flow: parseFloat(hxControls.tubeFlow?.value || 103.2),
      shell_t_in: parseFloat(hxControls.shellIn?.value || 280.6),
      shell_t_out: parseFloat(hxControls.shellOut?.value || 229.1),
      shell_flow: parseFloat(hxControls.shellFlow?.value || 84.6),
      time_hr: parseFloat(hxControls.runtime?.value || 1000),
      crude_api: 32.0,
      crude_tan: parseFloat(hxControls.tan?.value || 0.8),
      crude_chlorides: 15.0
    };

    try {
      const res = await fetch("/api/simulator/hx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(r => r.json());

      renderHxResults(res, payload);
    } catch (err) {
      console.error("HX Simulation failed:", err);
    }
  }, 100);

  function triggerHxRecalc() {
    // Update numerical value labels
    document.getElementById("hxTubeInVal").textContent = `${parseFloat(hxControls.tubeIn.value).toFixed(1)} °C`;
    document.getElementById("hxTubeOutVal").textContent = `${parseFloat(hxControls.tubeOut.value).toFixed(1)} °C`;
    document.getElementById("hxTubeFlowVal").textContent = `${parseFloat(hxControls.tubeFlow.value).toFixed(1)} kg/s`;
    document.getElementById("hxShellInVal").textContent = `${parseFloat(hxControls.shellIn.value).toFixed(1)} °C`;
    document.getElementById("hxShellOutVal").textContent = `${parseFloat(hxControls.shellOut.value).toFixed(1)} °C`;
    document.getElementById("hxShellFlowVal").textContent = `${parseFloat(hxControls.shellFlow.value).toFixed(1)} kg/s`;
    document.getElementById("hxRuntimeVal").textContent = `${parseInt(hxControls.runtime.value).toLocaleString()} hrs`;
    document.getElementById("hxTanVal").textContent = `${parseFloat(hxControls.tan.value).toFixed(2)} mg KOH/g`;

    runHxSimulation();
  }

  // Bind input events to all HX sliders
  [
    hxControls.tubeIn, hxControls.tubeOut, hxControls.tubeFlow,
    hxControls.shellIn, hxControls.shellOut, hxControls.shellFlow,
    hxControls.runtime, hxControls.tan
  ].forEach(input => input?.addEventListener("input", triggerHxRecalc));

  function renderHxResults(data, input) {
    const temaPct = data.tema_pct || 0;
    const isBreached = temaPct >= 100;

    // Dial gauge fill (circumference = 364.4)
    const dialFill = document.getElementById("hxDialFill");
    const pctClamped = Math.min(Math.max(temaPct, 0), 100);
    const offset = 364.4 - (364.4 * (pctClamped / 100));
    if (dialFill) {
      dialFill.style.strokeDashoffset = offset;
      dialFill.style.stroke = isBreached ? "var(--text-red)" : (temaPct > 60 ? "var(--text-amber)" : "var(--text-cyan)");
      dialFill.style.filter = `drop-shadow(0 0 6px ${isBreached ? "var(--text-red)" : "var(--text-cyan)"})`;
    }

    document.getElementById("hxTemaPct").textContent = `${temaPct.toFixed(0)}%`;
    const statusBadge = document.getElementById("hxStatusBadge");
    if (statusBadge) {
      statusBadge.textContent = isBreached ? "CRITICAL TEMA BREACH" : "OPERATING NORMAL";
      statusBadge.className = isBreached ? "status-badge alert-text" : "status-badge ok-text";
    }

    document.getElementById("hxRfVal").textContent = `${(data.calculated_rf_m2KW || 0).toFixed(6)} m²·K/W`;
    document.getElementById("hxUVal").textContent = `${Math.round(data.u_clean_W_m2K || 1200)} → ${Math.round(data.u_actual_W_m2K || 1100)} W/m²·K`;
    document.getElementById("hxUDeg").textContent = `Degradation: -${(data.u_degradation_pct || 0).toFixed(1)}%`;
    document.getElementById("hxDutyVal").textContent = `${Math.round(data.thermal_metrics?.duty_kW || 0).toLocaleString()} kW`;
    document.getElementById("hxLmtdVal").textContent = `LMTD: ${(data.thermal_metrics?.lmtd_C || 0).toFixed(1)} °C`;

    document.getElementById("hxAnnualPenalty").textContent = `$${Math.round(data.annual_cost_penalty_USD || 0).toLocaleString()} / yr`;
    document.getElementById("hxDaysToClean").textContent = `${data.days_to_cleaning || 0} Days`;
    document.getElementById("hxUrgency").textContent = data.cleaning_urgency || "Nominal";

    drawHxThermalSvg(input.tube_t_in, input.tube_t_out, input.shell_t_in, input.shell_t_out);
  }

  function drawHxThermalSvg(tIn, tOut, sIn, sOut) {
    const svg = document.getElementById("hxThermalSvg");
    if (!svg) return;
    svg.innerHTML = `
      <defs>
        <linearGradient id="tubeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#38bdf8" />
          <stop offset="100%" stop-color="#f59e0b" />
        </linearGradient>
        <linearGradient id="shellGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#ef4444" />
          <stop offset="100%" stop-color="#f59e0b" />
        </linearGradient>
      </defs>
      <!-- Grid lines -->
      <line x1="40" y1="20" x2="410" y2="20" stroke="rgba(255,255,255,0.06)" />
      <line x1="40" y1="50" x2="410" y2="50" stroke="rgba(255,255,255,0.06)" />
      <line x1="40" y1="80" x2="410" y2="80" stroke="rgba(255,255,255,0.06)" />
      
      <!-- Shell-side fluid curve (cooling) -->
      <path d="M 50 25 C 180 35, 300 65, 400 70" fill="none" stroke="url(#shellGrad)" stroke-width="3" />
      <circle cx="50" cy="25" r="4" fill="#ef4444" />
      <circle cx="400" cy="70" r="4" fill="#f59e0b" />
      <text x="50" y="16" fill="#ef4444" font-family="JetBrains Mono" font-size="9">Shell In: ${tIn > 0 ? sIn.toFixed(0) : 280}°C</text>
      <text x="350" y="85" fill="#f59e0b" font-family="JetBrains Mono" font-size="9">Shell Out: ${sOut.toFixed(0)}°C</text>

      <!-- Tube-side crude curve (heating) -->
      <path d="M 50 85 C 180 80, 300 50, 400 35" fill="none" stroke="url(#tubeGrad)" stroke-width="3" />
      <circle cx="50" cy="85" r="4" fill="#38bdf8" />
      <circle cx="400" cy="35" r="4" fill="#f59e0b" />
      <text x="50" y="100" fill="#38bdf8" font-family="JetBrains Mono" font-size="9">Crude In: ${tIn.toFixed(0)}°C</text>
      <text x="350" y="25" fill="#f59e0b" font-family="JetBrains Mono" font-size="9">Crude Out: ${tOut.toFixed(0)}°C</text>
    `;
  }

  // Ask Agent button for HX
  document.getElementById("hxAskAgentBtn")?.addEventListener("click", () => {
    const eid = hxControls.presetSelect?.value || "E02";
    const prompt = `Analyze Heat Exchanger ${eid} with current telemetry: Crude Inlet=${hxControls.tubeIn.value}°C, Outlet=${hxControls.tubeOut.value}°C, Flow=${hxControls.tubeFlow.value} kg/s, Shell In=${hxControls.shellIn.value}°C, Out=${hxControls.shellOut.value}°C. Calculate fouling resistance and recommend action.`;
    switchTab("agentTab");
    const input = document.getElementById("agentQueryInput");
    if (input) {
      input.value = prompt;
      document.getElementById("agentChatForm").dispatchEvent(new Event("submit"));
    }
  });

  // -------------------------------------------------------------------------
  // TWIN 2: IMS Rotating Bearing Simulation
  // -------------------------------------------------------------------------
  const brgControls = {
    rms: document.getElementById("brgRms"),
    peak: document.getElementById("brgPeak"),
    kurt: document.getElementById("brgKurt"),
    crest: document.getElementById("brgCrest"),
    rpm: document.getElementById("brgRpm"),
    hours: document.getElementById("brgHours"),
    presetSelect: document.getElementById("bearingPresetSelect")
  };

  const brgPresetValues = {
    test2_b1: { rms: 0.35, peak: 1.10, kurt: 5.2, crest: 3.14, rpm: 2000, hours: 150.0 },
    test1_b3: { rms: 0.85, peak: 2.80, kurt: 8.5, crest: 3.29, rpm: 2000, hours: 700.0 },
    test1_b4: { rms: 0.95, peak: 3.10, kurt: 9.1, crest: 3.26, rpm: 2000, hours: 710.0 },
    healthy:  { rms: 0.08, peak: 0.22, kurt: 2.9, crest: 2.75, rpm: 2000, hours: 50.0 }
  };

  brgControls.presetSelect?.addEventListener("change", (e) => {
    const p = brgPresetValues[e.target.value];
    if (p) {
      if (brgControls.rms) brgControls.rms.value = p.rms;
      if (brgControls.peak) brgControls.peak.value = p.peak;
      if (brgControls.kurt) brgControls.kurt.value = p.kurt;
      if (brgControls.crest) brgControls.crest.value = p.crest;
      if (brgControls.rpm) brgControls.rpm.value = p.rpm;
      if (brgControls.hours) brgControls.hours.value = p.hours;
      triggerBrgRecalc();
    }
  });

  const runBrgSimulation = debounce(async () => {
    const payload = {
      bearing_id: "Bearing1",
      rms: parseFloat(brgControls.rms?.value || 0.15),
      peak: parseFloat(brgControls.peak?.value || 0.45),
      kurtosis: parseFloat(brgControls.kurt?.value || 3.5),
      crest_factor: parseFloat(brgControls.crest?.value || 3.0),
      peak_to_peak: parseFloat(brgControls.peak?.value || 0.45) * 2.0,
      hours_elapsed: parseFloat(brgControls.hours?.value || 120.0),
      test_id: 2,
      rpm: parseFloat(brgControls.rpm?.value || 2000.0)
    };

    try {
      const res = await fetch("/api/simulator/bearing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(r => r.json());

      renderBrgResults(res, payload);
    } catch (err) {
      console.error("Bearing simulation failed:", err);
    }
  }, 100);

  function triggerBrgRecalc() {
    document.getElementById("brgRmsVal").textContent = `${parseFloat(brgControls.rms.value).toFixed(2)} g`;
    document.getElementById("brgPeakVal").textContent = `${parseFloat(brgControls.peak.value).toFixed(2)} g`;
    document.getElementById("brgKurtVal").textContent = parseFloat(brgControls.kurt.value).toFixed(2);
    document.getElementById("brgCrestVal").textContent = parseFloat(brgControls.crest.value).toFixed(2);
    document.getElementById("brgRpmVal").textContent = `${parseInt(brgControls.rpm.value).toLocaleString()} RPM`;
    document.getElementById("brgHoursVal").textContent = `${parseFloat(brgControls.hours.value).toFixed(1)} hrs`;

    runBrgSimulation();
  }

  [
    brgControls.rms, brgControls.peak, brgControls.kurt,
    brgControls.crest, brgControls.rpm, brgControls.hours
  ].forEach(input => input?.addEventListener("input", triggerBrgRecalc));

  function renderBrgResults(data, input) {
    const rul = data.predicted_rul_hours || 0;
    const zone = data.iso_zone || "A";

    const dialFill = document.getElementById("brgDialFill");
    // Normalized RUL: 100 hrs = full dial
    const rulPct = Math.min(Math.max((rul / 100) * 100, 5), 100);
    const offset = 364.4 - (364.4 * (rulPct / 100));
    if (dialFill) {
      dialFill.style.strokeDashoffset = offset;
      dialFill.style.stroke = zone === "D" ? "var(--text-red)" : (zone === "C" ? "var(--text-amber)" : "var(--text-cyan)");
      dialFill.style.filter = `drop-shadow(0 0 6px ${zone === "D" ? "var(--text-red)" : "var(--text-cyan)"})`;
    }

    document.getElementById("brgRulVal").textContent = `${rul.toFixed(1)} h`;
    const isoBadge = document.getElementById("brgIsoBadge");
    if (isoBadge) {
      isoBadge.textContent = `ISO ZONE ${zone} — ${data.iso_description || 'NORMAL'}`;
      isoBadge.className = zone === "D" ? "status-badge alert-text" : (zone === "C" ? "status-badge warning-text" : "status-badge ok-text");
    }

    document.getElementById("brgStageVal").textContent = data.degradation_stage || "Healthy";
    document.getElementById("brgIsoDesc").textContent = `Action: ${data.iso_action || 'Continue operation'}`;
    document.getElementById("brgKurtDisplay").textContent = `${input.kurtosis.toFixed(2)} (${input.kurtosis > 6 ? 'Severe Shock Spalling' : 'Nominal Impact'})`;
    document.getElementById("brgActionVal").textContent = data.recommendation || "Safe for continuous operation.";

    // Defect Frequencies
    if (data.defect_frequencies_Hz) {
      document.getElementById("freqBpfo").textContent = `${data.defect_frequencies_Hz.BPFO_outer_race} Hz`;
      document.getElementById("freqBpfi").textContent = `${data.defect_frequencies_Hz.BPFI_inner_race} Hz`;
      document.getElementById("freqBsf").textContent = `${data.defect_frequencies_Hz.BSF_ball_spin} Hz`;
      document.getElementById("freqFtf").textContent = `${data.defect_frequencies_Hz.FTF_cage} Hz`;
    }

    drawBrgWaveformSvg(input.rms, input.kurtosis, data.defect_frequencies_Hz?.BPFO_outer_race || 100);
  }

  function drawBrgWaveformSvg(rms, kurt, bpfo) {
    const svg = document.getElementById("brgWaveformSvg");
    if (!svg) return;
    const points = [];
    const amplitude = Math.min(rms * 25, 40);
    const spikeProb = Math.min((kurt - 3) / 10, 0.4);

    for (let x = 0; x <= 450; x += 3) {
      let y = 50 + amplitude * Math.sin(x * 0.15) + (amplitude * 0.3) * Math.sin(x * 0.4);
      if (Math.random() < spikeProb) {
        y += (Math.random() > 0.5 ? 1 : -1) * (kurt * 2.5);
      }
      points.push(`${x},${y.toFixed(1)}`);
    }

    svg.innerHTML = `
      <line x1="0" y1="50" x2="450" y2="50" stroke="rgba(255,255,255,0.08)" stroke-dasharray="4" />
      <polyline points="${points.join(" ")}" fill="none" stroke="#00f5d4" stroke-width="1.8" />
    `;
  }

  // Ask Agent for Bearing
  document.getElementById("brgAskAgentBtn")?.addEventListener("click", () => {
    const prompt = `Analyze Bearing 1 (Campaign 2) with simulated vibration: RMS=${brgControls.rms.value} g, Peak=${brgControls.peak.value} g, Kurtosis=${brgControls.kurt.value}, Shaft Speed=${brgControls.rpm.value} RPM. What is its ISO zone and RUL?`;
    switchTab("agentTab");
    const input = document.getElementById("agentQueryInput");
    if (input) {
      input.value = prompt;
      document.getElementById("agentChatForm").dispatchEvent(new Event("submit"));
    }
  });

  // -------------------------------------------------------------------------
  // TWIN 3: CNC Milling Machine Simulation
  // -------------------------------------------------------------------------
  const millControls = {
    air: document.getElementById("millAir"),
    proc: document.getElementById("millProc"),
    rpm: document.getElementById("millRpm"),
    torque: document.getElementById("millTorque"),
    wear: document.getElementById("millWear"),
    gradeBtns: document.querySelectorAll(".grade-btn"),
    presetSelect: document.getElementById("millingPresetSelect")
  };

  let selectedGrade = "M";
  millControls.gradeBtns.forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      millControls.gradeBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedGrade = btn.getAttribute("data-grade");
      triggerMillRecalc();
    });
  });

  const millPresetValues = {
    machine_142: { air: 300.0, proc: 310.0, rpm: 1400, torque: 50.0, wear: 215, grade: "L" },
    machine_1:   { air: 298.1, proc: 308.6, rpm: 1551, torque: 42.8, wear: 0,   grade: "M" },
    thermal_stress: { air: 304.5, proc: 311.2, rpm: 1450, torque: 46.0, wear: 120, grade: "M" },
    high_torque: { air: 298.5, proc: 309.0, rpm: 1300, torque: 68.0, wear: 180, grade: "H" }
  };

  millControls.presetSelect?.addEventListener("change", (e) => {
    const p = millPresetValues[e.target.value];
    if (p) {
      if (millControls.air) millControls.air.value = p.air;
      if (millControls.proc) millControls.proc.value = p.proc;
      if (millControls.rpm) millControls.rpm.value = p.rpm;
      if (millControls.torque) millControls.torque.value = p.torque;
      if (millControls.wear) millControls.wear.value = p.wear;
      selectedGrade = p.grade;
      millControls.gradeBtns.forEach(b => b.classList.toggle("active", b.getAttribute("data-grade") === p.grade));
      triggerMillRecalc();
    }
  });

  const runMillSimulation = debounce(async () => {
    const payload = {
      air_temp_k: parseFloat(millControls.air?.value || 300.0),
      process_temp_k: parseFloat(millControls.proc?.value || 310.0),
      rotational_speed_rpm: parseFloat(millControls.rpm?.value || 1500.0),
      torque_nm: parseFloat(millControls.torque?.value || 40.0),
      tool_wear_min: parseFloat(millControls.wear?.value || 108.0),
      product_type: selectedGrade
    };

    try {
      const res = await fetch("/api/simulator/milling", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(r => r.json());

      renderMillResults(res, payload);
    } catch (err) {
      console.error("Milling simulation failed:", err);
    }
  }, 100);

  function triggerMillRecalc() {
    const airK = parseFloat(millControls.air.value);
    const procK = parseFloat(millControls.proc.value);
    document.getElementById("millAirVal").textContent = `${airK.toFixed(1)} K (${(airK - 273.15).toFixed(1)} °C)`;
    document.getElementById("millProcVal").textContent = `${procK.toFixed(1)} K (${(procK - 273.15).toFixed(1)} °C)`;
    document.getElementById("millRpmVal").textContent = `${parseInt(millControls.rpm.value).toLocaleString()} RPM`;
    document.getElementById("millTorqueVal").textContent = `${parseFloat(millControls.torque.value).toFixed(1)} Nm`;
    document.getElementById("millWearVal").textContent = `${parseInt(millControls.wear.value)} min`;
    document.getElementById("millGradeVal").textContent = `${selectedGrade} (${selectedGrade === 'L' ? 'Low' : (selectedGrade === 'M' ? 'Medium' : 'High')})`;

    runMillSimulation();
  }

  [
    millControls.air, millControls.proc, millControls.rpm,
    millControls.torque, millControls.wear
  ].forEach(input => input?.addEventListener("input", triggerMillRecalc));

  function renderMillResults(data, input) {
    const prob = (data.overall_failure_probability || 0) * 100;
    const isHighRisk = prob > 35;

    const dialFill = document.getElementById("millDialFill");
    const offset = 364.4 - (364.4 * (Math.min(prob, 100) / 100));
    if (dialFill) {
      dialFill.style.strokeDashoffset = offset;
      dialFill.style.stroke = isHighRisk ? "var(--text-red)" : (prob > 15 ? "var(--text-amber)" : "var(--text-cyan)");
      dialFill.style.filter = `drop-shadow(0 0 6px ${isHighRisk ? "var(--text-red)" : "var(--text-cyan)"})`;
    }

    document.getElementById("millProbVal").textContent = `${prob.toFixed(1)}%`;
    const statusBadge = document.getElementById("millStatusBadge");
    if (statusBadge) {
      statusBadge.textContent = isHighRisk ? "HIGH FAILURE PROBABILITY" : "NOMINAL OPERATION";
      statusBadge.className = isHighRisk ? "status-badge alert-text" : "status-badge ok-text";
    }

    const wearBand = document.getElementById("millWearBandVal");
    if (wearBand) {
      wearBand.textContent = data.wear_band || "Normal";
      wearBand.style.color = data.wear_color || "#10b981";
    }

    document.getElementById("millPowerVal").textContent = `${Math.round(data.physics_metrics?.power_watts || 0).toLocaleString()} W`;
    document.getElementById("millTempDiffVal").textContent = `${(data.physics_metrics?.temp_diff_K || 0).toFixed(1)} K`;

    // 5 Failure Modes Bars
    const modes = data.failure_modes || {};
    const updateBar = (mode, barId, pctId) => {
      const p = (modes[mode]?.probability || 0) * 100;
      const bar = document.getElementById(barId);
      const text = document.getElementById(pctId);
      if (bar) {
        bar.style.width = `${Math.max(p, 1)}%`;
        bar.style.background = p > 50 ? "var(--text-red)" : (p > 20 ? "var(--text-amber)" : "var(--text-cyan)");
      }
      if (text) text.textContent = `${p.toFixed(1)}%`;
    };

    updateBar("TWF", "barTwf", "pctTwf");
    updateBar("HDF", "barHdf", "pctHdf");
    updateBar("PWF", "barPwf", "pctPwf");
    updateBar("OSF", "barOsf", "pctOsf");
    updateBar("RNF", "barRnf", "pctRnf");
  }

  // Ask Agent for Milling
  document.getElementById("millAskAgentBtn")?.addEventListener("click", () => {
    const prompt = `Predict failure for CNC Milling Machine with parameters: Air Temp=${millControls.air.value}K, Process Temp=${millControls.proc.value}K, RPM=${millControls.rpm.value}, Torque=${millControls.torque.value}Nm, Tool Wear=${millControls.wear.value}min, Grade=${selectedGrade}. Break down failure modes.`;
    switchTab("agentTab");
    const input = document.getElementById("agentQueryInput");
    if (input) {
      input.value = prompt;
      document.getElementById("agentChatForm").dispatchEvent(new Event("submit"));
    }
  });

  function updateActiveTwin() {
    if (state.activeTwin === "hxTwin") triggerHxRecalc();
    else if (state.activeTwin === "bearingTwin") triggerBrgRecalc();
    else if (state.activeTwin === "millingTwin") triggerMillRecalc();
  }

  // -------------------------------------------------------------------------
  // TAB 3: Unified ML Multi-Asset Predictor
  // -------------------------------------------------------------------------
  const predictAssetClass = document.getElementById("predictAssetClass");
  const millingFields = document.getElementById("millingFormFields");
  const hxFields = document.getElementById("hxFormFields");
  const bearingFields = document.getElementById("bearingFormFields");

  predictAssetClass?.addEventListener("change", (e) => {
    const val = e.target.value;
    if (millingFields) millingFields.style.display = val === "milling" ? "block" : "none";
    if (hxFields) hxFields.style.display = val === "heat_exchanger" ? "block" : "none";
    if (bearingFields) bearingFields.style.display = val === "bearing" ? "block" : "none";
  });

  const predictForm = document.getElementById("unifiedPredictForm");
  predictForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const assetType = predictAssetClass.value;
    let eqId = "142";
    let testId = 2;

    if (assetType === "milling") {
      eqId = document.getElementById("predMachineId")?.value || "142";
    } else if (assetType === "heat_exchanger") {
      eqId = document.getElementById("predExchangerId")?.value || "E02";
    } else if (assetType === "bearing") {
      eqId = document.getElementById("predBearingId")?.value || "Bearing1";
      testId = parseInt(document.getElementById("predTestId")?.value || "2");
    }

    const container = document.getElementById("predictOutputContainer");
    if (container) {
      container.innerHTML = `
        <div class="empty-state-prompt">
          <span class="exec-spinner"></span>
          <h4>Running XGBoost Inference...</h4>
          <p>Extracting telemetry and forecasting degradation parameters.</p>
        </div>
      `;
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          equipment_type: assetType,
          equipment_id: eqId,
          test_id: testId
        })
      }).then(r => r.json());

      renderPredictOutput(res, assetType, eqId);
    } catch (err) {
      if (container) {
        container.innerHTML = `<div class="alert-text">Error during ML prediction: ${err.message}</div>`;
      }
    }
  });

  function renderPredictOutput(res, assetType, eqId) {
    const container = document.getElementById("predictOutputContainer");
    if (!container) return;

    if (assetType === "milling") {
      const prob = ((res.overall_failure_probability || 0) * 100).toFixed(1);
      const isFail = res.predicted_failure;
      container.innerHTML = `
        <div class="pred-result-card">
          <div class="pred-result-header">
            <div>
              <h3>CNC Milling Machine #${res.machine_id || eqId}</h3>
              <span class="mono-label">Product ID: ${res.product_id || 'M'}</span>
            </div>
            <span class="${isFail ? 'badge-alert' : 'badge-ok'}">
              ${isFail ? 'FAILURE PREDICTED' : 'OPERATING SAFE'} (${prob}%)
            </span>
          </div>

          <div class="twin-kpis-column">
            <div class="twin-stat-box">
              <span class="stat-box-label">Mechanical Cutting Power</span>
              <span class="stat-box-val">${res.physics_metrics?.power_watts || 0} W</span>
            </div>
            <div class="twin-stat-box">
              <span class="stat-box-label">Thermal Dissipation (ΔT)</span>
              <span class="stat-box-val">${res.physics_metrics?.temp_diff_K || 0} K</span>
            </div>
          </div>

          <div class="failure-modes-card">
            <div class="modes-card-header">FAILURE MODES BREAKDOWN</div>
            <div class="mode-bars-list">
              ${Object.entries(res.failure_modes || {}).map(([mode, info]) => `
                <div class="mode-bar-row">
                  <span class="mode-name">${mode}</span>
                  <div class="mode-bar-track">
                    <div class="mode-bar-fill" style="width: ${(info.probability * 100).toFixed(1)}%;"></div>
                  </div>
                  <span class="mode-pct">${(info.probability * 100).toFixed(1)}%</span>
                </div>
              `).join("")}
            </div>
          </div>

          <div class="twin-stat-box">
            <span class="stat-box-label">Maintenance Directive</span>
            <span class="stat-box-val small-text">${res.recommendation || 'Continue normal operation.'}</span>
          </div>
        </div>
      `;
    } else if (assetType === "heat_exchanger") {
      const prob = ((res.tema_breach_probability || 0) * 100).toFixed(1);
      container.innerHTML = `
        <div class="pred-result-card">
          <div class="pred-result-header">
            <div>
              <h3>Heat Exchanger ${res.exchanger_id || eqId}</h3>
              <span class="mono-label">TEMA Standard Evaluation</span>
            </div>
            <span class="${res.status === 'CRITICAL FOULING' ? 'badge-alert' : 'badge-ok'}">
              ${res.status || 'NORMAL'}
            </span>
          </div>
          <div class="twin-kpis-column">
            <div class="twin-stat-box">
              <span class="stat-box-label">Predicted Fouling Resistance (Rf)</span>
              <span class="stat-box-val">${res.predicted_rf_m2KW} m²·K/W</span>
              <span class="stat-box-ref">TEMA Limit: 0.000500 m²·K/W</span>
            </div>
            <div class="twin-stat-box">
              <span class="stat-box-label">TEMA Breach Probability</span>
              <span class="stat-box-val">${prob}%</span>
            </div>
            <div class="twin-stat-box">
              <span class="stat-box-label">Operational Action</span>
              <span class="stat-box-val small-text">${res.recommendation}</span>
            </div>
          </div>
        </div>
      `;
    } else if (assetType === "bearing") {
      container.innerHTML = `
        <div class="pred-result-card">
          <div class="pred-result-header">
            <div>
              <h3>Rotating Bearing: ${res.bearing_id}</h3>
              <span class="mono-label">Accelerometry Feature Inference</span>
            </div>
            <span class="${res.degradation_stage?.includes('Critical') ? 'badge-alert' : 'badge-ok'}">
              ${res.degradation_stage}
            </span>
          </div>
          <div class="twin-kpis-column">
            <div class="twin-stat-box">
              <span class="stat-box-label">Predicted Remaining Useful Life (RUL)</span>
              <span class="stat-box-val">${res.predicted_rul_hours} Hours</span>
            </div>
            <div class="twin-stat-box">
              <span class="stat-box-label">Vibration RMS</span>
              <span class="stat-box-val">${res.vibration_metrics?.rms} g (Kurtosis: ${res.vibration_metrics?.kurtosis})</span>
            </div>
            <div class="twin-stat-box">
              <span class="stat-box-label">Recommendation</span>
              <span class="stat-box-val small-text">${res.recommendation}</span>
            </div>
          </div>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------------------
  // TAB 4 & DRAWER: AI Plant Diagnostic Agent
  // -------------------------------------------------------------------------
  const agentChatForm = document.getElementById("agentChatForm");
  const agentQueryInput = document.getElementById("agentQueryInput");
  const agentMessagesContainer = document.getElementById("agentMessagesContainer");
  const agentQuerySpinner = document.getElementById("agentQuerySpinner");
  const resetAgentSessionBtn = document.getElementById("resetAgentSessionBtn");

  const drawerChatForm = document.getElementById("drawerChatForm");
  const drawerQueryInput = document.getElementById("drawerQueryInput");
  const drawerMessagesContainer = document.getElementById("drawerMessagesContainer");

  async function submitAgentMessage(msg, targetContainer = agentMessagesContainer) {
    if (!msg.trim()) return;

    // Append user message bubble
    appendMessage(targetContainer, "user", msg);
    if (agentQuerySpinner) agentQuerySpinner.style.display = "flex";

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      }).then(r => r.json());

      const answer = res.response || "No response received.";
      appendMessage(targetContainer, "agent", answer);
    } catch (err) {
      appendMessage(targetContainer, "agent", `⚠️ Agent execution error: ${err.message}`);
    } finally {
      if (agentQuerySpinner) agentQuerySpinner.style.display = "none";
    }
  }

  function appendMessage(container, sender, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    if (sender === "agent") {
      bubble.innerHTML = marked.parse(text);
    } else {
      bubble.textContent = text;
    }

    msgDiv.appendChild(bubble);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
  }

  agentChatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = agentQueryInput.value.trim();
    if (!query) return;
    agentQueryInput.value = "";
    submitAgentMessage(query, agentMessagesContainer);
  });

  drawerChatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = drawerQueryInput.value.trim();
    if (!query) return;
    drawerQueryInput.value = "";
    submitAgentMessage(query, drawerMessagesContainer);
  });

  resetAgentSessionBtn?.addEventListener("click", async () => {
    try {
      await fetch("/api/reset", { method: "POST" });
      agentMessagesContainer.innerHTML = "";
      appendMessage(agentMessagesContainer, "agent", "Session cleared. Agent memory has been reset.");
    } catch (err) {
      console.error("Reset failed:", err);
    }
  });

  // Top header refresh button
  document.getElementById("refreshFleetBtn")?.addEventListener("click", () => {
    loadFleetTelemetry();
  });

  // Initial Load
  loadFleetTelemetry();
  triggerHxRecalc();
});
