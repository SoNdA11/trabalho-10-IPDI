/* Inicialização do Reveal, tema, rodapé e números vindos dos dados reais. */
(function () {
  "use strict";
  const printing = /print-pdf/i.test(window.location.search);
  const D = window.DADOS;

  // ─── números nos slides: <span data-v="eurosat.test.accuracy" data-f="pct"> ───
  const get = (path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), D);
  const fmt = {
    pct: (v) => (v * 100).toFixed(1).replace(".", ",") + "%",
    pct0: (v) => Math.round(v * 100) + "%",
    dec2: (v) => Number(v).toFixed(2).replace(".", ","),
    dec3: (v) => Number(v).toFixed(3).replace(".", ","),
    dec1: (v) => Number(v).toFixed(1).replace(".", ","),
    int: (v) => Math.round(v).toLocaleString("pt-BR"),
    km2: (v) => Number(v).toFixed(1).replace(".", ",") + " km²",
    pp: (v) => (v * 100 >= 0 ? "+" : "") + (v * 100).toFixed(1).replace(".", ",") + " p.p.",
  };
  document.querySelectorAll("[data-v]").forEach((el) => {
    let v = get(el.dataset.v);
    if (el.dataset.minus) v = v - get(el.dataset.minus);
    if (v == null) { el.textContent = "?"; el.style.color = "red"; return; }
    el.textContent = (fmt[el.dataset.f] || String)(v);
  });

  // ─── tema: claro por padrão; ?tema=escuro ou T alternam (lembrado) ───
  const root = document.documentElement;
  const stored = () => { try { return localStorage.getItem("trab10-slides-tema"); } catch (e) { return null; } };
  function setTheme(t) {
    if (t === "escuro") root.setAttribute("data-tema", "escuro"); else root.removeAttribute("data-tema");
    try { localStorage.setItem("trab10-slides-tema", t); } catch (e) { /* sem armazenamento */ }
  }
  const qs = new URLSearchParams(location.search).get("tema");
  setTheme(qs || (printing ? "claro" : stored()) || "claro");

  Reveal.initialize({
    width: 1920, height: 1080, margin: 0, minScale: 0.2, maxScale: 2.0,
    center: false, hash: true, controls: false, progress: true, slideNumber: false,
    transition: "fade", transitionSpeed: "fast", backgroundTransition: "none",
    pdfSeparateFragments: false, defaultTiming: 50,
    plugins: [RevealNotes],
  });

  Reveal.addKeyBinding({ keyCode: 84, key: "T", description: "Alternar tema claro/escuro" }, () => {
    setTheme(root.getAttribute("data-tema") === "escuro" ? "claro" : "escuro");
    window.DeckCharts.retheme();
  });

  const footer = document.querySelector(".deck-footer");
  const num = document.getElementById("footer-num");
  const who = document.getElementById("footer-who");
  const total = document.querySelectorAll(".reveal .slides > section").length;
  const secColor = { s1: "--sec1", s2: "--sec2", s3: "--sec3", s4: "--sec4", s5: "--sec5" };

  function update(ev) {
    const slide = ev.currentSlide;
    const h = Reveal.getIndices(slide).h;
    num.textContent = `${h + 1} / ${total}`;
    who.textContent = slide.dataset.who || "";
    const sec = Object.keys(secColor).find((s) => slide.classList.contains(s));
    root.style.setProperty("--sec-cur", sec ? `var(${secColor[sec]})` : "var(--teal)");
    footer.style.visibility = slide.classList.contains("capa") ? "hidden" : "visible";
    window.DeckCharts.onSlide(slide);
  }

  Reveal.on("ready", (ev) => {
    if (printing) { window.DeckCharts.buildAll(); return; }
    update(ev);
  });
  Reveal.on("slidechanged", update);
  window.addEventListener("resize", () => {
    document.querySelectorAll(".chart[id]").forEach((el) => { const c = echarts.getInstanceByDom(el); if (c) c.resize(); });
  });
})();
