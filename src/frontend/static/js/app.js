/**
 * Aplicación Frontend Principal para el Dashboard Analítico de la Premier League.
 * Consume los endpoints de FastAPI (/api/v1/...) de forma 100% desacoplada.
 */

const API_BASE = ""; // Origen relativo al backend de FastAPI

// Estado de la aplicación
const state = {
  activeTab: "overview",
  sessionId: localStorage.getItem("epl_session_id") || crypto.randomUUID(),
  teams: [],
  selectedSeason: 2024,
  charts: {}
};

// Guardar sessionId en localStorage
localStorage.setItem("epl_session_id", state.sessionId);

// ==========================================================================
// INICIALIZACIÓN
// ==========================================================================
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupChatHandlers();
  setupPredictionsHandlers();
  setupTeamHandlers();
  setupStorytellingHandlers();

  // Carga inicial de datos
  await loadTeamsList();
  await loadOverview();
  await loadStandings(state.selectedSeason);
});

// ==========================================================================
// NAVEGACIÓN ENTRE SECCIONES
// ==========================================================================
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      switchTab(target);
    });
  });
}

function switchTab(targetTab) {
  state.activeTab = targetTab;
  
  // Actualizar botones de navegación
  document.querySelectorAll(".nav-tab").forEach(tab => {
    if (tab.dataset.tab === targetTab) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  // Mostrar la sección correspondiente
  document.querySelectorAll(".tab-content").forEach(sec => {
    if (sec.id === `section-${targetTab}`) {
      sec.classList.remove("hidden");
    } else {
      sec.classList.add("hidden");
    }
  });

  // Acciones específicas al entrar a una pestaña
  if (targetTab === "overview") {
    loadOverview();
  } else if (targetTab === "standings") {
    loadStandings(state.selectedSeason);
  }
}

// ==========================================================================
// SECCIÓN A: OVERVIEW / INICIO
// ==========================================================================
async function loadOverview() {
  try {
    const [overviewRes, seasonsRes] = await Promise.all([
      fetch(`${API_BASE}/api/v1/analytics/overview`).then(r => r.json()),
      fetch(`${API_BASE}/api/v1/analytics/seasons`).then(r => r.json())
    ]);

    if (overviewRes.status === "success") {
      document.getElementById("kpi-matches").textContent = overviewRes.total_matches.toLocaleString();
      document.getElementById("kpi-goals").textContent = overviewRes.total_goals.toLocaleString();
      document.getElementById("kpi-avg-goals").textContent = overviewRes.avg_goals_per_match.toFixed(2);
      document.getElementById("kpi-teams").textContent = overviewRes.teams_count;
      document.getElementById("kpi-home-wins").textContent = `${overviewRes.home_wins_pct}%`;
      document.getElementById("kpi-draws").textContent = `${overviewRes.draws_pct}%`;
      document.getElementById("kpi-away-wins").textContent = `${overviewRes.away_wins_pct}%`;

      renderOutcomeDonutChart(overviewRes);
    }

    if (seasonsRes.status === "success") {
      renderSeasonsBarChart(seasonsRes.seasons);
    }
  } catch (err) {
    console.error("Error cargando Overview:", err);
  }
}

function renderOutcomeDonutChart(data) {
  const ctx = document.getElementById("chart-outcome-donut");
  if (!ctx) return;

  if (state.charts.outcomeDonut) {
    state.charts.outcomeDonut.destroy();
  }

  state.charts.outcomeDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Victorias Local", "Empates", "Victorias Visitante"],
      datasets: [{
        data: [data.home_wins_pct, data.draws_pct, data.away_wins_pct],
        backgroundColor: ["#00ff87", "#38bdf8", "#f43f5e"],
        borderWidth: 2,
        borderColor: "#0f172a"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { size: 12 } }
        }
      },
      cutout: "65%"
    }
  });
}

function renderSeasonsBarChart(seasons) {
  const ctx = document.getElementById("chart-seasons-bar");
  if (!ctx) return;

  if (state.charts.seasonsBar) {
    state.charts.seasonsBar.destroy();
  }

  const labels = seasons.map(s => `${s.season}/${(s.season + 1).toString().slice(-2)}`);
  const goals = seasons.map(s => s.total_goals);
  const avg = seasons.map(s => s.avg_goals);

  state.charts.seasonsBar = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Total Goles",
          data: goals,
          backgroundColor: "rgba(56, 0, 60, 0.75)",
          borderColor: "#00ff87",
          borderWidth: 1.5,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
      },
      plugins: {
        legend: { labels: { color: "#94a3b8" } }
      }
    }
  });
}

// ==========================================================================
// SECCIÓN B: CLASIFICACIÓN / STANDINGS
// ==========================================================================
async function loadStandings(season) {
  const tbody = document.getElementById("standings-tbody");
  tbody.innerHTML = `<tr><td colspan="11" class="text-center py-8 text-slate-400">Cargando clasificación de la temporada ${season}...</td></tr>`;

  try {
    const res = await fetch(`${API_BASE}/api/v1/standings?season=${season}`).then(r => r.json());
    if (res.status !== "success") {
      tbody.innerHTML = `<tr><td colspan="11" class="text-center py-8 text-rose-400">${res.detail || "Error al consultar la tabla"}</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    res.tabla.forEach(row => {
      let badgeClass = "cluster-badge-midtable";
      if (row.cluster === 0) badgeClass = "cluster-badge-europe";
      else if (row.cluster === 2) badgeClass = "cluster-badge-relegation";

      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/40 transition-colors";
      tr.innerHTML = `
        <td class="py-3 px-3 text-center font-bold text-slate-300">${row.posicion}</td>
        <td class="py-3 px-3 font-semibold text-white flex items-center gap-2">
          <span>${row.team_name}</span>
        </td>
        <td class="py-3 px-2 text-center text-slate-400">${row.partidos_jugados}</td>
        <td class="py-3 px-2 text-center font-bold text-emerald-400">${row.puntos}</td>
        <td class="py-3 px-2 text-center text-slate-300">${row.victorias}</td>
        <td class="py-3 px-2 text-center text-slate-300">${row.empates}</td>
        <td class="py-3 px-2 text-center text-slate-300">${row.derrotas}</td>
        <td class="py-3 px-2 text-center text-slate-300">${row.goles_favor}</td>
        <td class="py-3 px-2 text-center text-slate-300">${row.goles_contra}</td>
        <td class="py-3 px-2 text-center font-medium ${row.diferencia_goles >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
          ${row.diferencia_goles > 0 ? '+' : ''}${row.diferencia_goles}
        </td>
        <td class="py-3 px-3 text-center">
          <span class="inline-block px-2.5 py-0.5 text-xs rounded-full font-medium ${badgeClass}">
            ${row.etiqueta_cluster}
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="11" class="text-center py-8 text-rose-400">Error de conexión al cargar la clasificación.</td></tr>`;
  }
}

// Selector de temporada en standings
document.getElementById("standings-season-select")?.addEventListener("change", (e) => {
  state.selectedSeason = parseInt(e.target.value);
  loadStandings(state.selectedSeason);
});

// ==========================================================================
// SECCIÓN C: EQUIPOS Y COMPARADOR
// ==========================================================================
async function loadTeamsList() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/teams`).then(r => r.json());
    if (res.status === "success") {
      state.teams = res.teams;
      populateTeamSelectors();
    }
  } catch (err) {
    console.error("Error al cargar lista de clubes:", err);
  }
}

function populateTeamSelectors() {
  const selects = [
    document.getElementById("team-profile-select"),
    document.getElementById("compare-team-a"),
    document.getElementById("compare-team-b"),
    document.getElementById("pred-home-team"),
    document.getElementById("pred-away-team")
  ];

  selects.forEach((sel, idx) => {
    if (!sel) return;
    sel.innerHTML = '<option value="">-- Seleccionar Equipo --</option>';
    state.teams.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t.team_name;
      opt.textContent = t.team_name;
      sel.appendChild(opt);
    });
  });

  // Valores por defecto
  if (document.getElementById("team-profile-select")) document.getElementById("team-profile-select").value = "Manchester City";
  if (document.getElementById("compare-team-a")) document.getElementById("compare-team-a").value = "Arsenal";
  if (document.getElementById("compare-team-b")) document.getElementById("compare-team-b").value = "Manchester City";
  if (document.getElementById("pred-home-team")) document.getElementById("pred-home-team").value = "Arsenal";
  if (document.getElementById("pred-away-team")) document.getElementById("pred-away-team").value = "Liverpool";

  // Cargar perfil por defecto
  loadTeamProfile("Manchester City");
}

function setupTeamHandlers() {
  document.getElementById("team-profile-select")?.addEventListener("change", (e) => {
    if (e.target.value) loadTeamProfile(e.target.value);
  });

  document.getElementById("btn-compare-teams")?.addEventListener("click", () => {
    const teamA = document.getElementById("compare-team-a").value;
    const teamB = document.getElementById("compare-team-b").value;
    const season = document.getElementById("compare-season-select").value;
    if (teamA && teamB) loadTeamComparison(teamA, teamB, season);
  });
}

async function loadTeamProfile(teamName) {
  const container = document.getElementById("team-profile-details");
  try {
    const res = await fetch(`${API_BASE}/api/v1/teams/${encodeURIComponent(teamName)}/profile`).then(r => r.json());
    if (res.status !== "success") return;

    const p = res.ultimo_perfil;
    let badgeClass = "cluster-badge-midtable";
    if (p.cluster === 0) badgeClass = "cluster-badge-europe";
    else if (p.cluster === 2) badgeClass = "cluster-badge-relegation";

    container.innerHTML = `
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50">
          <div class="text-xs text-slate-400">Puntos (2024)</div>
          <div class="text-2xl font-bold text-emerald-400">${p.puntos}</div>
          <div class="text-xs text-slate-400">${p.victorias}V - ${p.empates}E - ${p.derrotas}D</div>
        </div>
        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50">
          <div class="text-xs text-slate-400">Goles a Favor</div>
          <div class="text-2xl font-bold text-sky-400">${p.goles_favor}</div>
          <div class="text-xs text-slate-400">Prom: ${(p.goles_favor/38).toFixed(2)} por partido</div>
        </div>
        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50">
          <div class="text-xs text-slate-400">Goles Recibidos</div>
          <div class="text-2xl font-bold text-rose-400">${p.goles_contra}</div>
          <div class="text-xs text-slate-400">Diferencia: ${p.diferencia_goles > 0 ? '+' : ''}${p.diferencia_goles}</div>
        </div>
        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50">
          <div class="text-xs text-slate-400">Clúster K-Means</div>
          <div class="text-sm font-semibold mt-1">
            <span class="inline-block px-2.5 py-1 text-xs rounded-full font-medium ${badgeClass}">
              ${p.etiqueta_cluster}
            </span>
          </div>
        </div>
      </div>
    `;

    renderTeamHistoryChart(res.perfiles_historicos);
  } catch (err) {
    console.error("Error al cargar perfil:", err);
  }
}

function renderTeamHistoryChart(history) {
  const ctx = document.getElementById("chart-team-history");
  if (!ctx) return;

  if (state.charts.teamHistory) state.charts.teamHistory.destroy();

  const sorted = [...history].reverse();
  const labels = sorted.map(h => `Temporada ${h.season}`);
  const pts = sorted.map(h => h.puntos);
  const gf = sorted.map(h => h.goles_favor);
  const ga = sorted.map(h => h.goles_contra);

  state.charts.teamHistory = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: "Puntos", data: pts, backgroundColor: "#00ff87", borderRadius: 4 },
        { label: "Goles a Favor", data: gf, backgroundColor: "#38bdf8", borderRadius: 4 },
        { label: "Goles Recibidos", data: ga, backgroundColor: "#f43f5e", borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
      },
      plugins: {
        legend: { position: "top", labels: { color: "#94a3b8" } }
      }
    }
  });
}

async function loadTeamComparison(teamA, teamB, season) {
  const container = document.getElementById("team-comparison-results");
  container.innerHTML = `<div class="text-center py-6 text-slate-400">Comparando ${teamA} vs ${teamB} (Temporada ${season})...</div>`;

  try {
    const res = await fetch(`${API_BASE}/api/v1/teams/compare?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}&season=${season}`).then(r => r.json());
    if (res.status !== "success") {
      container.innerHTML = `<div class="text-center py-4 text-rose-400">${res.detail || "Error en la comparación"}</div>`;
      return;
    }

    const a = res.team_a;
    const b = res.team_b;

    container.innerHTML = `
      <div class="grid grid-cols-3 gap-2 text-center text-sm py-2 border-b border-slate-700/60 font-semibold text-slate-300">
        <div class="text-sky-400">${a.team_name}</div>
        <div class="text-xs uppercase text-slate-400">Métrica (${season})</div>
        <div class="text-emerald-400">${b.team_name}</div>
      </div>
      <div class="divide-y divide-slate-800/60 text-sm">
        <div class="grid grid-cols-3 gap-2 text-center py-2.5">
          <span class="font-bold text-white">${a.puntos}</span>
          <span class="text-slate-400 text-xs">Puntos Totales</span>
          <span class="font-bold text-white">${b.puntos}</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2.5">
          <span class="text-slate-300">${a.victorias}V / ${a.empates}E / ${a.derrotas}D</span>
          <span class="text-slate-400 text-xs">Récord V - E - D</span>
          <span class="text-slate-300">${b.victorias}V / ${b.empates}E / ${b.derrotas}D</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2.5">
          <span class="text-sky-300 font-semibold">${a.goles_favor}</span>
          <span class="text-slate-400 text-xs">Goles a Favor</span>
          <span class="text-sky-300 font-semibold">${b.goles_favor}</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2.5">
          <span class="text-rose-300 font-semibold">${a.goles_contra}</span>
          <span class="text-slate-400 text-xs">Goles en Contra</span>
          <span class="text-rose-300 font-semibold">${b.goles_contra}</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2.5">
          <span class="${a.diferencia_goles >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-bold">${a.diferencia_goles > 0 ? '+' : ''}${a.diferencia_goles}</span>
          <span class="text-slate-400 text-xs">Diferencia de Goles</span>
          <span class="${b.diferencia_goles >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-bold">${b.diferencia_goles > 0 ? '+' : ''}${b.diferencia_goles}</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2.5">
          <span class="text-xs text-emerald-400 font-medium">${a.etiqueta_cluster}</span>
          <span class="text-slate-400 text-xs">Clúster K-Means</span>
          <span class="text-xs text-emerald-400 font-medium">${b.etiqueta_cluster}</span>
        </div>
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<div class="text-center py-4 text-rose-400">Error conectando con la API de comparación.</div>`;
  }
}

// ==========================================================================
// SECCIÓN D: PREDICCIONES DE PARTIDOS (1X2)
// ==========================================================================
function setupPredictionsHandlers() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const homeTeam = document.getElementById("pred-home-team").value;
    const awayTeam = document.getElementById("pred-away-team").value;
    const modelType = document.getElementById("pred-model-select").value;

    if (!homeTeam || !awayTeam) {
      alert("Por favor selecciona tanto el equipo local como el visitante.");
      return;
    }
    if (homeTeam === awayTeam) {
      alert("El equipo local y el visitante no pueden ser el mismo.");
      return;
    }

    const btn = document.getElementById("btn-run-prediction");
    const resultBox = document.getElementById("prediction-result-box");
    
    btn.disabled = true;
    btn.innerHTML = `<span class="inline-block animate-spin mr-2">⏳</span> Evaluando modelo supervisado...`;
    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `<div class="text-center py-8 text-slate-400">Calculando probabilidades estadísticas...</div>`;

    try {
      const res = await fetch(`${API_BASE}/api/v1/predictions/match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          model_type: modelType
        })
      }).then(r => r.json());

      if (res.status !== "success") {
        resultBox.innerHTML = `<div class="p-4 bg-rose-900/30 border border-rose-700/50 rounded-xl text-rose-300">${res.detail || "Error al generar predicción"}</div>`;
        return;
      }

      const probs = res.probabilidades;
      const homePct = (probs.Local * 100).toFixed(1);
      const drawPct = (probs.Empate * 100).toFixed(1);
      const awayPct = (probs.Visita * 100).toFixed(1);

      let outcomeBadgeClass = "bg-sky-500/20 text-sky-300 border-sky-500/30";
      if (res.prediccion_estimada === "Local") outcomeBadgeClass = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
      if (res.prediccion_estimada === "Visita") outcomeBadgeClass = "bg-rose-500/20 text-rose-300 border-rose-500/30";

      const factorsHtml = res.factores_clave.map(f => `<li class="flex items-start gap-2"><span class="text-emerald-400">✓</span><span>${f}</span></li>`).join("");

      resultBox.innerHTML = `
        <div class="p-6 bg-slate-900/80 border border-slate-700/60 rounded-2xl">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800">
            <div>
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-1">Encuentro Evaluado</div>
              <h3 class="text-xl font-bold text-white">${res.home_team} <span class="text-slate-500 text-sm">vs</span> ${res.away_team}</h3>
              <div class="text-xs text-slate-400 mt-1">Modelo Operativo: <span class="text-sky-400">${res.modelo_utilizado}</span></div>
            </div>
            <div class="text-right">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-1">Resultado Más Probable</div>
              <span class="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold border ${outcomeBadgeClass}">
                ${res.prediccion_estimada} (${(res.confianza_maxima * 100).toFixed(1)}%)
              </span>
            </div>
          </div>

          <!-- Barras de probabilidad porcentual -->
          <div class="mb-6 space-y-3">
            <div>
              <div class="flex justify-between text-xs mb-1">
                <span class="text-slate-300 font-medium">Victoria Local (${res.home_team})</span>
                <span class="text-emerald-400 font-bold">${homePct}%</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                <div class="bg-emerald-400 h-3 rounded-full transition-all duration-700" style="width: ${homePct}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-xs mb-1">
                <span class="text-slate-300 font-medium">Empate</span>
                <span class="text-sky-400 font-bold">${drawPct}%</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                <div class="bg-sky-400 h-3 rounded-full transition-all duration-700" style="width: ${drawPct}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-xs mb-1">
                <span class="text-slate-300 font-medium">Victoria Visitante (${res.away_team})</span>
                <span class="text-rose-400 font-bold">${awayPct}%</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                <div class="bg-rose-400 h-3 rounded-full transition-all duration-700" style="width: ${awayPct}%"></div>
              </div>
            </div>
          </div>

          <!-- Factores clave -->
          <div class="mb-5 bg-slate-800/40 p-4 rounded-xl border border-slate-700/40">
            <h4 class="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Factores Estadísticos Clave</h4>
            <ul class="text-xs text-slate-300 space-y-1.5">${factorsHtml}</ul>
          </div>

          <!-- Advertencia Metodológica -->
          <div class="text-[11px] text-amber-300/80 bg-amber-950/20 border border-amber-800/30 p-3 rounded-xl">
            <span class="font-bold text-amber-400">⚠️ Advertencia Metodológica:</span> ${res.advertencia_metodologica}
          </div>
        </div>
      `;
    } catch (err) {
      resultBox.innerHTML = `<div class="p-4 bg-rose-900/30 border border-rose-700/50 rounded-xl text-rose-300">Error de red al conectar con el servidor predictivo.</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>⚡</span><span>Ejecutar Predicción</span>`;
    }
  });
}

// ==========================================================================
// SECCIÓN E: ASISTENTE IA (CHAT)
// ==========================================================================
function setupChatHandlers() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const messagesBox = document.getElementById("chat-messages");
  const resetBtn = document.getElementById("btn-chat-reset");

  // Chips de preguntas rápidas
  document.querySelectorAll(".quick-prompt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      input.value = btn.textContent.trim().replace(/^["']|["']$/g, "");
      form.dispatchEvent(new Event("submit"));
    });
  });

  resetBtn?.addEventListener("click", async () => {
    if (!confirm("¿Deseas reiniciar la memoria de la conversación actual?")) return;
    try {
      await fetch(`${API_BASE}/api/v1/chat/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId })
      });
      messagesBox.innerHTML = `
        <div class="flex items-start gap-3">
          <div class="w-8 h-8 rounded-full bg-purple-900/60 border border-emerald-400/30 flex items-center justify-center text-xs text-emerald-400 font-bold shrink-0">PL</div>
          <div class="chat-bubble-bot p-4 rounded-2xl max-w-[85%] text-sm">
            <p class="font-medium text-emerald-400 mb-1">Antigravity Sports Analyst</p>
            <p>Memoria reiniciada. ¿Qué consulta o pronóstico deseas analizar ahora?</p>
          </div>
        </div>
      `;
    } catch (err) {
      console.error("Error al reiniciar chat:", err);
    }
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    appendUserMessage(query);

    // Mensaje de carga
    const loaderId = "loader-" + Date.now();
    appendLoaderMessage(loaderId);

    try {
      const res = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: query,
          session_id: state.sessionId
        })
      }).then(r => r.json());

      removeMessage(loaderId);

      if (res.status === "error") {
        appendBotMessage(`**Error en el agente:** ${res.response || "No se pudo procesar la consulta."}`, []);
      } else {
        appendBotMessage(res.response, res.tool_calls || [], res.api_notice);
      }
    } catch (err) {
      removeMessage(loaderId);
      appendBotMessage("Error de conexión al contactar al agente analítico.", []);
    }
  });
}

function appendUserMessage(text) {
  const box = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "flex justify-end";
  div.innerHTML = `
    <div class="chat-bubble-user p-4 rounded-2xl max-w-[80%] text-sm">
      <p class="whitespace-pre-wrap">${escapeHtml(text)}</p>
    </div>
  `;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function appendLoaderMessage(id) {
  const box = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.id = id;
  div.className = "flex items-start gap-3";
  div.innerHTML = `
    <div class="w-8 h-8 rounded-full bg-purple-900/60 border border-emerald-400/30 flex items-center justify-center text-xs text-emerald-400 font-bold shrink-0">PL</div>
    <div class="chat-bubble-bot p-4 rounded-2xl text-sm flex items-center gap-2 text-slate-400">
      <span class="inline-block animate-spin">⏳</span>
      <span>Consultando herramientas y razonando...</span>
    </div>
  `;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendBotMessage(markdownText, toolCalls, apiNotice) {
  const box = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "flex items-start gap-3";

  // Chips de herramientas
  let toolsHtml = "";
  if (toolCalls && toolCalls.length > 0) {
    const badges = toolCalls.map(tc => `<span class="tool-badge">⚙️ ${tc.tool}</span>`).join(" ");
    toolsHtml = `<div class="flex flex-wrap gap-1.5 mb-2.5">${badges}</div>`;
  }

  // Notificación de cuota/modelo
  let noticeHtml = "";
  if (apiNotice) {
    noticeHtml = `<div class="mb-2 text-xs text-amber-300/80 bg-amber-950/30 p-2 rounded border border-amber-800/40">ℹ️ ${escapeHtml(apiNotice)}</div>`;
  }

  // Gráfico si la herramienta generó uno
  let chartHtml = "";
  if (toolCalls) {
    toolCalls.forEach(tc => {
      if (tc.output && tc.output.image_base64) {
        chartHtml += `<div class="my-3 rounded-xl overflow-hidden border border-slate-700 max-w-lg"><img src="data:image/png;base64,${tc.output.image_base64}" class="w-full h-auto" alt="Gráfico generado" /></div>`;
      }
    });
  }

  // Parsear Markdown y sanitizar con DOMPurify para prevenir vulnerabilidades XSS
  const rawHtml = typeof marked !== "undefined" ? marked.parse(markdownText) : escapeHtml(markdownText);
  const parsedContent = typeof DOMPurify !== "undefined" ? DOMPurify.sanitize(rawHtml, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'b', 'i', 'code', 'pre', 'blockquote',
      'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'table', 'thead', 'tbody', 'tr', 'th', 'td', 'span', 'hr', 'div'
    ],
    ALLOWED_ATTR: ['class'],
    FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover']
  }) : escapeHtml(markdownText);

  div.innerHTML = `
    <div class="w-8 h-8 rounded-full bg-purple-900/60 border border-emerald-400/30 flex items-center justify-center text-xs text-emerald-400 font-bold shrink-0">PL</div>
    <div class="chat-bubble-bot p-4 rounded-2xl max-w-[85%] text-sm">
      <div class="font-medium text-emerald-400 text-xs mb-1.5 flex items-center justify-between">
        <span>Antigravity Sports Analyst</span>
      </div>
      ${noticeHtml}
      ${toolsHtml}
      <div class="prose prose-invert prose-sm max-w-none space-y-2">
        ${parsedContent}
      </div>
      ${chartHtml}
    </div>
  `;

  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ==========================================================================
// SECCIÓN F: STORYTELLING
// ==========================================================================
function setupStorytellingHandlers() {
  document.querySelectorAll(".story-card-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const prompt = btn.dataset.prompt;
      if (!prompt) return;

      // Cambiar a la pestaña de chat y ejecutar la consulta
      switchTab("chat");
      const input = document.getElementById("chat-input");
      input.value = prompt;
      document.getElementById("chat-form").dispatchEvent(new Event("submit"));
    });
  });
}

// ==========================================================================
// UTILIDADES
// ==========================================================================
function escapeHtml(string) {
  const pre = document.createElement("pre");
  const text = document.createTextNode(string);
  pre.appendChild(text);
  return pre.innerHTML;
}
