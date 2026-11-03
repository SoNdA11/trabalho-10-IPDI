/* Gráficos ECharts alimentados por data/dados.js (resultados reais do projeto).
   Cada gráfico é montado quando o slide aparece e refeito ao trocar o tema. */
(function () {
  "use strict";
  const D = window.DADOS;
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const pct = (v, d = 1) => (v * 100).toFixed(d).replace(".", ",") + "%";
  const dec = (v, d = 2) => Number(v).toFixed(d).replace(".", ",");
  const LC = ["Vegetação", "Área agrícola", "Área urbana", "Solo exposto", "Água"];
  const LCVAR = { "Vegetação": "--veg", "Área agrícola": "--agr", "Área urbana": "--urb", "Solo exposto": "--solo", "Água": "--agua" };
  const lc = (c) => css(LCVAR[c]);
  const printing = /print-pdf/i.test(location.search);

  function base() {
    return {
      animation: !printing, animationDuration: 900,
      textStyle: { fontFamily: "Inter, sans-serif", color: css("--text"), fontSize: 20 },
      grid: { left: 90, right: 40, top: 50, bottom: 70, containLabel: true },
      tooltip: { show: false },
    };
  }
  const axis = (extra = {}) => Object.assign({
    axisLine: { lineStyle: { color: css("--line") } }, axisTick: { show: false },
    axisLabel: { color: css("--muted"), fontSize: 19 }, splitLine: { lineStyle: { color: css("--line"), opacity: 0.6 } },
    nameTextStyle: { color: css("--muted"), fontSize: 18 },
  }, extra);

  // ─── definições ───────────────────────────────────────────────────────
  const defs = {
    // assinaturas espectrais dos pixels de Mossoró (por classe de referência)
    "ch-assinaturas": () => {
      const s = D.sig_scene;
      return Object.assign(base(), {
        legend: { top: 0, textStyle: { color: css("--muted"), fontSize: 19 }, itemWidth: 26 },
        grid: { left: 80, right: 30, top: 70, bottom: 60, containLabel: true },
        xAxis: axis({
          type: "value", min: 400, max: 2300, name: "comprimento de onda (nm)", nameLocation: "middle", nameGap: 44,
          splitLine: { show: false }
        }),
        yAxis: axis({ type: "value", name: "reflectância", min: 0, max: 0.5, axisLabel: { color: css("--muted"), fontSize: 19, formatter: (v) => dec(v, 1) } }),
        series: LC.map((c) => ({
          name: c, type: "line", smooth: false, symbolSize: 10, lineStyle: { width: 4 },
          itemStyle: { color: lc(c) }, data: s.wavelength.map((w, i) => [w, s.mean[c][i]]),
          markArea: c === "Vegetação" ? {
            silent: true, itemStyle: { opacity: 0.07 }, label: { color: css("--faint"), fontSize: 17, position: "insideTop" },
            data: [[{ name: "VIS", xAxis: 400, itemStyle: { color: css("--blue") } }, { xAxis: 700 }],
            [{ name: "NIR", xAxis: 700, itemStyle: { color: css("--nir") } }, { xAxis: 900 }],
            [{ name: "SWIR", xAxis: 1500, itemStyle: { color: css("--swir") } }, { xAxis: 2300 }]]
          } : undefined,
        })),
      });
    },

    // proporção das classes na área de estudo (pixels de referência)
    "ch-proporcao": () => {
      const t = D.s2.class_pixels_total, tot = Object.values(t).reduce((a, b) => a + b, 0);
      return Object.assign(base(), {
        grid: { left: 20, right: 90, top: 10, bottom: 10, containLabel: true },
        xAxis: axis({ type: "value", max: 1, show: false }),
        yAxis: axis({ type: "category", inverse: true, data: LC, axisLabel: { color: css("--text"), fontSize: 22 } }),
        series: [{
          type: "bar", barWidth: 30, data: LC.map((c) => ({ value: t[c] / tot, itemStyle: { color: lc(c), borderRadius: 6 } })),
          label: { show: true, position: "right", color: css("--text"), fontSize: 21, fontFamily: "JetBrains Mono", formatter: (p) => pct(p.value) }
        }],
      });
    },

    // boxplots dos índices por classe (pixels de Mossoró)
    "ch-box-ndvi": () => box("NDVI"),
    "ch-box-mndwi": () => box("MNDWI"),
    "ch-box-ndbi": () => box("NDBI"),

    // importância das características (pixel, Sentinel-2)
    "ch-imp-s2": () => importance(D.s2.feature_importance.slice(0, 10), "--sec3"),
    "ch-imp-eu": () => importance(D.eurosat.feature_importance.slice(0, 10), "--sec3"),

    // matriz de confusão EuroSAT (normalizada por linha)
    "ch-cm-eu": () => cm(D.eurosat.test.confusion_matrix, D.eurosat.classes.map((c) => D.pt[c]), 15),
    "ch-cm-s2": () => cm(D.s2.test.confusion_matrix, LC, 20),

    // comparação de modelos e de divisões
    "ch-modelos": () => {
      const c = D.eurosat.comparisons;
      const rows = [["Random Forest — 13 bandas", "RF · 13 bandas"], ["SVM (RBF) — 13 bandas", "SVM · 13 bandas"],
      ["k-NN (k=7) — 13 bandas", "k-NN · 13 bandas"], ["Random Forest — só RGB", "RF · só RGB"]];
      return Object.assign(base(), {
        legend: { top: 0, right: 0, textStyle: { color: css("--muted"), fontSize: 19 } },
        grid: { left: 20, right: 30, top: 60, bottom: 20, containLabel: true },
        xAxis: axis({ type: "category", data: rows.map((r) => r[1]), axisLabel: { color: css("--text"), fontSize: 21 } }),
        yAxis: axis({ type: "value", min: 0.7, max: 1, axisLabel: { color: css("--muted"), fontSize: 18, formatter: (v) => pct(v, 0) } }),
        series: [
          {
            name: "acurácia", type: "bar", barGap: "12%", barWidth: 46, data: rows.map((r) => c[r[0]].accuracy),
            itemStyle: { color: css("--sec4"), borderRadius: [6, 6, 0, 0] },
            label: { show: true, position: "top", fontFamily: "JetBrains Mono", fontSize: 19, color: css("--text"), formatter: (p) => pct(p.value) }
          },
          {
            name: "F1-macro", type: "bar", barWidth: 46, data: rows.map((r) => c[r[0]].macro_f1),
            itemStyle: { color: css("--faint"), borderRadius: [6, 6, 0, 0] },
            label: { show: true, position: "insideTop", fontFamily: "JetBrains Mono", fontSize: 16, color: "#fff", formatter: (p) => pct(p.value) }
          },
        ],
      });
    },

    // F1 por classe: Sentinel-2 × CBERS-4
    "ch-f1-sensores": () => Object.assign(base(), {
      legend: { top: 0, right: 0, textStyle: { color: css("--muted"), fontSize: 20 } },
      grid: { left: 20, right: 20, top: 60, bottom: 20, containLabel: true },
      xAxis: axis({ type: "category", data: LC, axisLabel: { color: css("--text"), fontSize: 21 } }),
      yAxis: axis({ type: "value", min: 0, max: 1, axisLabel: { color: css("--muted"), fontSize: 18, formatter: (v) => dec(v, 1) } }),
      series: [
        {
          name: "Sentinel-2 · 13 bandas · 10 m", type: "bar", barWidth: 52, data: LC.map((c) => D.s2.test.per_class[c].f1),
          itemStyle: { color: css("--sec5"), borderRadius: [6, 6, 0, 0] },
          label: { show: true, position: "top", fontFamily: "JetBrains Mono", fontSize: 19, color: css("--text"), formatter: (p) => dec(p.value) }
        },
        {
          name: "CBERS-4 · 4 bandas · 20 m", type: "bar", barWidth: 52, data: LC.map((c) => D.cbers.test.per_class[c].f1),
          itemStyle: { color: css("--faint"), borderRadius: [6, 6, 0, 0] },
          label: { show: true, position: "top", fontFamily: "JetBrains Mono", fontSize: 17, color: css("--muted"), formatter: (p) => dec(p.value) }
        },
      ],
    }),

    // áreas (km²): mapa × referência
    "ch-areas": () => Object.assign(base(), {
      legend: { top: 0, right: 0, textStyle: { color: css("--muted"), fontSize: 19 } },
      grid: { left: 10, right: 80, top: 50, bottom: 10, containLabel: true },
      yAxis: axis({ type: "category", inverse: true, data: LC.slice(1), axisLabel: { color: css("--text"), fontSize: 21 } }),
      xAxis: axis({ type: "value", axisLabel: { color: css("--muted"), fontSize: 17 } }),
      series: [
        {
          name: "mapa do modelo", type: "bar", barWidth: 22, itemStyle: { color: css("--muted") }, data: LC.slice(1).map((c) => ({ value: D.s2.area_km2_pred[c], itemStyle: { color: lc(c) } })),
          label: { show: true, position: "right", fontFamily: "JetBrains Mono", fontSize: 17, color: css("--text"), formatter: (p) => dec(p.value, 1) }
        },
        {
          name: "referência", type: "bar", barWidth: 22, itemStyle: { color: css("--muted"), opacity: 0.35 }, data: LC.slice(1).map((c) => ({ value: D.s2.area_km2_ref[c], itemStyle: { color: lc(c), opacity: 0.35 } })),
          label: { show: true, position: "right", fontFamily: "JetBrains Mono", fontSize: 16, color: css("--muted"), formatter: (p) => dec(p.value, 1) }
        },
      ],
    }),
  };

  function box(index) {
    const ind = D.sig_scene.indices;
    const all = LC.flatMap((c) => ind[c][index]);
    const range = [Math.floor(Math.min(...all) * 10 - 0.5) / 10, Math.ceil(Math.max(...all) * 10 + 0.5) / 10];
    return Object.assign(base(), {
      title: { text: index, left: "center", top: 0, textStyle: { color: css("--text"), fontSize: 26, fontWeight: 700 } },
      grid: { left: 20, right: 20, top: 56, bottom: 10, containLabel: true },
      xAxis: axis({ type: "category", data: LC.map((c) => c.replace("Área ", "")), axisLabel: { color: css("--muted"), fontSize: 17, interval: 0 } }),
      yAxis: axis({ type: "value", min: range[0], max: range[1], axisLabel: { color: css("--muted"), fontSize: 16, formatter: (v) => dec(v, 1) } }),
      series: [{ type: "boxplot", boxWidth: [30, 46], data: LC.map((c) => ({ value: ind[c][index], itemStyle: { color: lc(c), borderColor: css("--text"), borderWidth: 1.5, opacity: 0.9 } })) }],
    });
  }

  function importance(list, colorVar) {
    const rows = list.slice().reverse();
    return Object.assign(base(), {
      grid: { left: 10, right: 80, top: 10, bottom: 10, containLabel: true },
      xAxis: axis({ type: "value", show: false }),
      yAxis: axis({ type: "category", data: rows.map((r) => r.name), axisLabel: { color: css("--text"), fontSize: 19, fontFamily: "JetBrains Mono" } }),
      series: [{
        type: "bar", barWidth: 24, data: rows.map((r) => r.importance), itemStyle: { color: css(colorVar), borderRadius: 5 },
        label: { show: true, position: "right", fontFamily: "JetBrains Mono", fontSize: 17, color: css("--muted"), formatter: (p) => dec(p.value, 3) }
      }],
    });
  }

  function cm(matrix, labels, fs) {
    const data = [];
    matrix.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((v, j) => data.push([j, i, v / tot, v]));
    });
    return Object.assign(base(), {
      grid: { left: 10, right: 20, top: 10, bottom: 10, containLabel: true },
      xAxis: axis({ type: "category", data: labels, position: "top", axisLabel: { color: css("--muted"), fontSize: fs, rotate: labels.length > 6 ? 35 : 0, interval: 0 }, splitLine: { show: false } }),
      yAxis: axis({ type: "category", data: labels, inverse: true, axisLabel: { color: css("--text"), fontSize: fs + 1, interval: 0 }, splitLine: { show: false } }),
      visualMap: { show: false, dimension: 2, min: 0, max: 1, inRange: { color: [css("--bg-2"), css("--sec4")] } },
      series: [{
        type: "heatmap", data, itemStyle: { borderColor: css("--bg"), borderWidth: 2, borderRadius: 4 },
        label: {
          show: true, fontFamily: "JetBrains Mono", fontSize: fs, formatter: (p) => (p.value[2] < 0.005 ? "" : Math.round(p.value[2] * 100)),
          color: css("--text")
        }
      }],
    });
  }

  // ─── espectro de bandas (SVG) ─────────────────────────────────────────
  function bandas(el) {
    const W = 1700, x = (nm) => 40 + ((nm - 400) / (2300 - 400)) * (W - 80);
    const s2 = [["B01", 443, 20, 60], ["B02", 490, 65, 10], ["B03", 560, 35, 10], ["B04", 665, 30, 10], ["B05", 705, 15, 20],
    ["B06", 740, 15, 20], ["B07", 783, 20, 20], ["B08", 842, 115, 10], ["B8A", 865, 20, 20], ["B09", 945, 20, 60],
    ["B10", 1375, 30, 60], ["B11", 1610, 90, 20], ["B12", 2190, 180, 20]];
    const mux = [["B5", 485, 70], ["B6", 555, 70], ["B7", 660, 60], ["B8", 830, 120]];
    const resColor = { 10: css("--sec2"), 20: css("--sec3"), 60: css("--faint") };
    const h = { 10: 120, 20: 90, 60: 60 };
    let svg = `<svg viewBox="0 0 ${W} 470" width="100%">`;
    // faixas do espectro
    [[400, 700, "visível", "--blue"], [700, 1000, "infravermelho próximo (NIR)", "--nir"], [1300, 2300, "infravermelho de ondas curtas (SWIR)", "--swir"]].forEach(([a, b, n, v]) => {
      svg += `<rect x="${x(a)}" y="0" width="${x(b) - x(a)}" height="300" fill="${css(v)}" opacity="0.07" rx="8"/>`;
      svg += `<text x="${(x(a) + x(b)) / 2}" y="30" text-anchor="middle" font-size="20" fill="${css("--muted")}" class="mono">${n}</text>`;
    });
    // gradiente do visível
    svg += `<defs><linearGradient id="vis" x1="0" x2="1"><stop offset="0" stop-color="#6a3df0"/><stop offset="0.25" stop-color="#2f6bd8"/><stop offset="0.5" stop-color="#2f9e44"/><stop offset="0.75" stop-color="#f2c12e"/><stop offset="1" stop-color="#e03131"/></linearGradient></defs>`;
    svg += `<rect x="${x(400)}" y="292" width="${x(700) - x(400)}" height="8" fill="url(#vis)" rx="3"/>`;
    s2.forEach(([n, c, bw, r], i) => {
      const hh = h[r], bx = x(c - bw / 2), wd = Math.max(x(c + bw / 2) - bx, 9);
      svg += `<rect class="pop" style="--delay:${0.2 + i * 0.06}s" x="${bx}" y="${290 - hh}" width="${wd}" height="${hh}" fill="${resColor[r]}" opacity="0.88" rx="3"/>`;
      const ty = 290 - hh - 10 - (["B05", "B07", "B8A"].includes(n) ? 26 : 0);
      svg += `<text class="pop mono" style="--delay:${0.2 + i * 0.06}s" x="${x(c)}" y="${ty}" text-anchor="middle" font-size="17" fill="${css("--text")}">${n}</text>`;
    });
    // CBERS MUX
    svg += `<text x="${x(400) - 10}" y="370" text-anchor="end" font-size="18" fill="${css("--muted")}"></text>`;
    mux.forEach(([n, c, bw], i) => {
      svg += `<rect class="pop" style="--delay:${1.2 + i * 0.1}s" x="${x(c - bw / 2)}" y="340" width="${x(c + bw / 2) - x(c - bw / 2)}" height="40" fill="${css("--sec5")}" opacity="0.85" rx="4"/>`;
      svg += `<text class="pop mono" style="--delay:${1.2 + i * 0.1}s" x="${x(c)}" y="366" text-anchor="middle" font-size="17" fill="#fff">${n}</text>`;
    });
    svg += `<text x="${x(1000)}" y="368" font-size="21" fill="${css("--sec5")}" font-weight="700">CBERS-4 MUX · 4 bandas · 20 m · sem SWIR</text>`;
    // eixo
    [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200].forEach((nm) => {
      svg += `<text x="${x(nm)}" y="425" text-anchor="middle" font-size="17" fill="${css("--faint")}" class="mono">${nm}</text>`;
    });
    svg += `<text x="${W / 2}" y="462" text-anchor="middle" font-size="19" fill="${css("--muted")}">comprimento de onda (nm)</text>`;
    svg += `</svg>`;
    el.innerHTML = svg;
  }

  // ─── ciclo de vida ────────────────────────────────────────────────────
  function build(el) {
    if (el.dataset.svg === "bandas") { bandas(el); return; }
    const fn = defs[el.id];
    if (!fn) return;
    let c = echarts.getInstanceByDom(el);
    if (c) c.dispose();
    c = echarts.init(el, null, { renderer: printing ? "svg" : "canvas" });
    c.setOption(fn());
  }

  window.DeckCharts = {
    onSlide(slide) { slide.querySelectorAll(".chart[id], [data-svg]").forEach(build); },
    buildAll() { document.querySelectorAll(".chart[id], [data-svg]").forEach(build); },
    retheme() { document.querySelectorAll(".chart[id], [data-svg]").forEach((el) => { const c = window.echarts && echarts.getInstanceByDom(el); if (c || el.dataset.svg) build(el); }); },
    pct, dec,
  };
})();
