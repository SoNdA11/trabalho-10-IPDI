/* Interface de demonstração — consome a API Flask (src/app.py).
   Nada aqui é resultado fixo: cada clique chama o classificador treinado. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const pct = (v, d = 1) => (v * 100).toFixed(d).replace(".", ",") + "%";
  const num = (v, d = 3) => Number(v).toFixed(d).replace(".", ",");
  const img = (b64) => "data:image/png;base64," + b64;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const PT = { AnnualCrop: "Cultura anual", Forest: "Floresta", HerbaceousVegetation: "Vegetação herbácea",
    Highway: "Rodovia", Industrial: "Industrial", Pasture: "Pastagem", PermanentCrop: "Cultura permanente",
    Residential: "Residencial", River: "Rio", SeaLake: "Mar / lago" };
  const EU_COLORS = { AnnualCrop: "#E3B505", Forest: "#1B7F3B", HerbaceousVegetation: "#7BC96F", Highway: "#8A8F98",
    Industrial: "#C2185B", Pasture: "#C9D86B", PermanentCrop: "#E07B39", Residential: "#7E57C2", River: "#3D8BFD", SeaLake: "#0D3B8C" };
  let LC_COLORS = {};

  async function api(url, opts) {
    const r = await fetch(url, opts);
    let data;
    try { data = await r.json(); } catch (e) { throw new Error(`resposta inválida (${r.status})`); }
    if (!r.ok || data.error) throw new Error(data.error || `erro ${r.status}`);
    return data;
  }

  // ─── tema (persistido) ──────────────────────────────────────────────────
  const root = document.documentElement;
  function setTheme(t) {
    if (t === "escuro") root.setAttribute("data-tema", "escuro"); else root.removeAttribute("data-tema");
    try { localStorage.setItem("trab10-tema", t); } catch (e) { /* sem armazenamento */ }
  }
  let saved = null;
  try { saved = localStorage.getItem("trab10-tema"); } catch (e) { saved = null; }
  setTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "escuro" : "claro"));
  const toggleTheme = () => setTheme(root.getAttribute("data-tema") === "escuro" ? "claro" : "escuro");
  $("theme").addEventListener("click", toggleTheme);
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "t" || e.key === "T") toggleTheme();
    if (["1", "2", "3"].includes(e.key)) showTab(["chip", "scene", "metrics"][+e.key - 1]);
  });

  // ─── abas ───────────────────────────────────────────────────────────────
  const loaded = {};
  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + name));
    if (!loaded[name]) { loaded[name] = true; ({ chip: initChip, scene: initScene, metrics: initMetrics })[name](); }
  }
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

  // ─── status ─────────────────────────────────────────────────────────────
  api("/api/status").then((s) => {
    const names = { rf_eurosat_ms: "EuroSAT 13b", rf_eurosat_rgb: "EuroSAT RGB", rf_s2_pixel: "Sentinel-2", rf_cbers_pixel: "CBERS-4" };
    $("status").innerHTML = Object.entries(s.models)
      .map(([k, ok]) => `<span class="pill ${ok ? "ok" : "no"}" title="${k}">${names[k] || k}</span>`).join("");
  }).catch(() => { $("status").innerHTML = '<span class="pill no">API indisponível</span>'; });

  function bars(entries, colorOf, fmt = pct) {
    return `<div class="bars">${entries.map(([k, v]) => `
      <div class="bar"><span>${esc(k)}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%;background:${colorOf(k)}"></div></div><span class="v">${fmt(v)}</span></div>`).join("")}</div>`;
  }

  // ═══ 1. Imagem inteira ═════════════════════════════════════════════════
  let samples = [];
  async function initChip() {
    const sel = $("classFilter");
    Object.entries(PT).forEach(([k, v]) => sel.insertAdjacentHTML("beforeend", `<option value="${k}">${v}</option>`));
    sel.addEventListener("change", renderSamples);
    try {
      samples = (await api("/api/eurosat/samples")).samples;
      renderSamples();
    } catch (e) { $("samples").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
    $("upload").addEventListener("change", uploadFile);
  }

  function renderSamples() {
    const f = $("classFilter").value;
    $("samples").innerHTML = samples.filter((s) => !f || s.class === f).map((s) => `
      <button data-id="${s.id}" title="${esc(s.class_pt)} · ${esc(s.file)}"><img src="${img(s.thumb)}" alt="${esc(s.class_pt)}"><span style="background:${EU_COLORS[s.class]}"></span></button>`).join("");
    $("samples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => classifySample(+b.dataset.id, b)));
  }

  async function classifySample(id, btn) {
    $("samples").querySelectorAll("button").forEach((b) => b.classList.toggle("sel", b === btn));
    $("chipResult").classList.add("loading");
    try { renderChip(await api(`/api/eurosat/classify/${id}`)); }
    catch (e) { $("chipResult").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
    $("chipResult").classList.remove("loading");
  }

  async function uploadFile(ev) {
    const f = ev.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    $("chipResult").classList.add("loading");
    try { renderChip(await api("/api/eurosat/upload", { method: "POST", body: fd }), f.name); }
    catch (e) { $("chipResult").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
    $("chipResult").classList.remove("loading");
    ev.target.value = "";
  }

  function renderChip(r, filename) {
    const views = Object.entries(r.views || {});
    const label = { cor_verdadeira: "cor verdadeira", falsa_cor: "falsa cor (NIR)", swir: "SWIR", ndvi: "NDVI" };
    const probs = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);
    const truth = r.true_class
      ? `<div>Classe real: <b>${esc(r.true_pt)}</b> · <span class="${r.correct ? "okc" : "badc"}">${r.correct ? "✓ acerto" : "✗ erro"}</span></div>`
      : `<div class="hint">${esc(filename || "")} · ${esc(r.input || "")}</div>`;
    const idx = r.indices ? `<div class="kpis">${Object.entries(r.indices).map(([k, v]) =>
      `<div class="kpi"><span class="big">${num(v, 2)}</span><small>${k} médio</small></div>`).join("")}</div>` : "";
    $("chipResult").innerHTML = `
      <div class="panel-head"><h2>Resultado</h2><span class="meta">${esc(r.model)} · ${r.n_features} características · ${num(r.inference_ms, 1)} ms</span></div>
      <div class="views ${views.length === 1 ? "one" : ""}">${views.map(([k, v]) => `<figure><img src="${img(v)}" alt=""><figcaption>${label[k] || k}</figcaption></figure>`).join("")}</div>
      <div class="verdict">
        <span class="badge" style="background:${r.color}">${esc(r.class_pt)}</span>
        <span class="big">${pct(r.confidence)}</span>
        <span class="hint">macro-classe: <b>${esc(r.macro_class)}</b></span>
      </div>
      ${truth}
      ${idx}
      <div class="grid2">
        <div><h2>Probabilidades (votos das árvores)</h2>${bars(probs.map(([k, v]) => [PT[k], v]), (k) => EU_COLORS[Object.keys(PT).find((c) => PT[c] === k)])}</div>
        <div><h2>Características mais importantes</h2>
          <table><thead><tr><th>característica</th><th class="num">valor</th><th class="num">import.</th></tr></thead>
          <tbody>${r.features.map((f) => `<tr><td>${esc(f.name)}</td><td class="num">${num(f.value, 4)}</td><td class="num">${num(f.importance, 3)}</td></tr>`).join("")}</tbody></table>
        </div>
      </div>`;
  }

  // ═══ 2. Cena do INPE (por pixel) ═══════════════════════════════════════
  let scenes = [], scene = null, lastClick = null;
  async function initScene() {
    try {
      const d = await api("/api/scenes");
      scenes = d.scenes;
      d.classes.forEach((c) => { LC_COLORS[c.name] = c.color; });
      $("legend").innerHTML = d.classes.map((c) => `<span style="--c:${c.color}">${esc(c.name)}</span>`).join("");
    } catch (e) { $("sceneWrap").innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    $("sceneSel").innerHTML = scenes.map((s) => `<button data-id="${s.id}">${s.satellite} · ${s.sensor}</button>`).join("");
    $("sceneSel").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => selectScene(b.dataset.id)));
    $("sceneImg").addEventListener("click", sceneClick);
    $("btnFull").addEventListener("click", classifyFull);
    $("winSize").addEventListener("change", () => lastClick && runWindow(lastClick));
    $("majority").addEventListener("change", () => lastClick && runWindow(lastClick));
    if (scenes.length) selectScene(scenes[0].id);
  }

  function selectScene(id) {
    scene = scenes.find((s) => s.id === id);
    $("sceneSel").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
    $("sceneImg").src = img(scene.truecolor);
    $("sceneWrap").classList.remove("show-map");
    $("fullInfo").textContent = "";
    $("sceneMeta").textContent = `${scene.date} · ${scene.level} · ${scene.bands} bandas · ${scene.resolution_m} m · índices: ${scene.indices.join(", ")}`;
    const c = { x: Math.round(scene.width * 0.42), y: Math.round(scene.height * 0.5) };
    runWindow(c);
    queryPixel(c);
  }

  function sceneClick(ev) {
    const r = ev.target.getBoundingClientRect();
    const c = { x: Math.round(((ev.clientX - r.left) / r.width) * scene.width),
                y: Math.round(((ev.clientY - r.top) / r.height) * scene.height) };
    runWindow(c);
    queryPixel(c);
  }

  function drawBox(x, y, size) {
    const b = $("winBox");
    b.style.display = "block";
    b.style.left = (x / scene.width) * 100 + "%";
    b.style.top = (y / scene.height) * 100 + "%";
    b.style.width = (size / scene.width) * 100 + "%";
    b.style.height = (size / scene.height) * 100 + "%";
  }

  async function runWindow(c) {
    lastClick = c;
    const size = +$("winSize").value;
    const body = { x: c.x - size / 2, y: c.y - size / 2, size, majority: $("majority").checked };
    $("winMeta").textContent = "classificando…";
    try {
      const r = await api(`/api/scenes/${scene.id}/window`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      drawBox(r.x, r.y, r.size);
      $("wTrue").src = img(r.truecolor); $("wMap").src = img(r.map); $("wRef").src = img(r.reference);
      const km = ((r.size * r.resolution_m) / 1000).toFixed(2).replace(".", ",");
      $("winMeta").textContent = `${r.n_pixels.toLocaleString("pt-BR")} pixels · ${km} km · ${num(r.ms, 0)} ms · concordância ${r.agreement == null ? "—" : pct(r.agreement)}`;
      $("winFractions").innerHTML = bars(Object.entries(r.fractions), (k) => LC_COLORS[k]);
    } catch (e) { $("winMeta").textContent = e.message; }
  }

  async function queryPixel(c) {
    try {
      const r = await api(`/api/scenes/${scene.id}/pixel?x=${c.x}&y=${c.y}`);
      if (!r.valid) { $("pixelInfo").innerHTML = `<div class="empty small">(${r.x}, ${r.y}): ${esc(r.reason)}</div>`; $("pxMeta").textContent = ""; return; }
      $("pxMeta").textContent = `(${r.x}, ${r.y}) · ${num(r.ms, 1)} ms`;
      const match = r.reference === r.class_name;
      $("pixelInfo").innerHTML = `
        <div class="verdict"><span class="badge" style="background:${r.color}">${esc(r.class_name)}</span>
          <span class="hint">referência: <b>${esc(r.reference || "sem classe")}</b> (${esc(r.worldcover)}) ${r.reference ? `<span class="${match ? "okc" : "badc"}">${match ? "✓" : "✗"}</span>` : ""}</span></div>
        <div class="two">
          <div>${bars(Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]), (k) => LC_COLORS[k])}</div>
          <div class="feat-list"><table><thead><tr><th>característica</th><th class="num">valor</th></tr></thead>
            <tbody>${r.features.map((f) => `<tr><td>${esc(f.name)}</td><td class="num">${num(f.value, 4)}</td></tr>`).join("")}</tbody></table></div>
        </div>`;
    } catch (e) { $("pixelInfo").innerHTML = `<div class="empty small">${esc(e.message)}</div>`; }
  }

  async function classifyFull() {
    const btn = $("btnFull");
    btn.disabled = true;
    const t0 = performance.now();
    btn.textContent = "classificando todos os pixels…";
    try {
      const r = await api(`/api/scenes/${scene.id}/full`, { method: "POST" });
      $("fullMap").src = img(r.map);
      $("sceneWrap").classList.add("show-map");
      const area = Object.entries(r.area_km2).map(([k, v]) => `${k}: ${v.toFixed(1).replace(".", ",")} km²`).join(" · ");
      $("fullInfo").innerHTML = `<b>${r.n_pixels.toLocaleString("pt-BR")} pixels</b> classificados agora em ${num(r.seconds, 1)} s (ida e volta ${num((performance.now() - t0) / 1000, 1)} s) · concordância com a referência ${pct(r.agreement)}<br>${area} · <a href="#" id="toggleMap">alternar mapa/imagem</a>`;
      $("toggleMap").addEventListener("click", (e) => { e.preventDefault(); $("sceneWrap").classList.toggle("show-map"); });
    } catch (e) { $("fullInfo").textContent = e.message; }
    btn.disabled = false;
    btn.textContent = "Classificar a cena inteira";
  }

  // ═══ 3. Métricas ═══════════════════════════════════════════════════════
  async function initMetrics() {
    let d;
    try { d = await api("/api/metrics"); } catch (e) { $("metrics").innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    const title = { eurosat: "Imagem inteira — EuroSAT MS (10 classes)", s2_pixel: "Por pixel — Sentinel-2 L1C, Mossoró/RN", cbers_pixel: "Por pixel — CBERS-4 MUX, Mossoró/RN" };
    let html = "";
    for (const k of ["eurosat", "s2_pixel", "cbers_pixel"]) {
      const m = d[k];
      if (!m) continue;
      const names = m.classes.map((c) => PT[c] || c);
      const cm = m.confusion_matrix;
      const rows = cm.map((row, i) => {
        const tot = row.reduce((a, b) => a + b, 0) || 1;
        return `<tr><th>${esc(names[i])}</th>${row.map((v, j) => {
          const p = v / tot;
          const bg = i === j ? `rgba(10,158,136,${0.15 + p * 0.85})` : `rgba(214,56,74,${Math.min(p * 3, 0.9)})`;
          return `<td style="background:${bg};color:${p > 0.5 ? "#fff" : "inherit"}" title="${v}">${(p * 100).toFixed(0)}</td>`;
        }).join("")}</tr>`;
      }).join("");
      html += `<div class="panel mcard">
        <div class="panel-head"><h2>${title[k]}</h2><span class="meta">${esc(m.split)}</span></div>
        <div class="kpis">
          <div class="kpi"><span class="big">${pct(m.accuracy)}</span><small>acurácia</small></div>
          <div class="kpi"><span class="big">${num(m.kappa, 3)}</span><small>kappa</small></div>
          <div class="kpi"><span class="big">${pct(m.macro_f1)}</span><small>F1-macro</small></div>
          <div class="kpi"><span class="big">${m.n_test.toLocaleString("pt-BR")}</span><small>amostras de teste</small></div>
        </div>
        <div class="grid2">
          <div><table><thead><tr><th>classe</th><th class="num">precisão</th><th class="num">recall</th><th class="num">F1</th><th class="num">suporte</th></tr></thead><tbody>
            ${Object.entries(m.per_class).map(([c, v]) => `<tr><td>${esc(PT[c] || c)}</td><td class="num">${pct(v.precision)}</td><td class="num">${pct(v.recall)}</td><td class="num">${pct(v.f1)}</td><td class="num">${v.support.toLocaleString("pt-BR")}</td></tr>`).join("")}
          </tbody></table></div>
          <div class="cm"><table><thead><tr><th>real ↓ / previsto →</th>${names.map((n) => `<th>${esc(n.slice(0, 6))}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>
          <div class="hint">% de cada linha (classe real). Diagonal = acertos.</div></div>
        </div>
        ${m.comparisons ? `<h2 class="mt">Comparações no mesmo teste</h2><table><thead><tr><th>modelo</th><th class="num">acurácia</th><th class="num">F1-macro</th><th class="num">treino (s)</th></tr></thead><tbody>
          ${Object.entries(m.comparisons).map(([n, v]) => `<tr><td>${esc(n)}</td><td class="num">${pct(v.accuracy)}</td><td class="num">${pct(v.macro_f1)}</td><td class="num">${num(v.train_seconds, 1)}</td></tr>`).join("")}
          <tr><td>Random Forest — divisão aleatória (vazamento)</td><td class="num">${pct(m.random_split_test.accuracy)}</td><td class="num">${pct(m.random_split_test.macro_f1)}</td><td class="num">—</td></tr>
        </tbody></table>` : ""}
      </div>`;
    }
    if (d.s2_region) {
      const r = d.s2_region;
      html += `<div class="panel mcard"><div class="panel-head"><h2>Por região — modelo EuroSAT aplicado em Mossoró (janelas de 640 m)</h2></div>
        <div class="kpis"><div class="kpi"><span class="big">${pct(r.agreement_macro)}</span><small>concordância (4 macro-classes)</small></div>
        <div class="kpi"><span class="big">${r.n_windows.toLocaleString("pt-BR")}</span><small>janelas</small></div>
        <div class="kpi"><span class="big">${num(r.seconds, 1)} s</span><small>tempo</small></div></div>
        <p class="hint">${esc(r.note)}</p></div>`;
    }
    $("metrics").innerHTML = html;
  }

  showTab("chip");
})();
