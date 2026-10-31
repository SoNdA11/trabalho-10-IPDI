/* ============================================================
   slides.js — Trabalho 10 (navegação, tema e helpers)
   ============================================================ */
(function () {
  'use strict';
  const KEY = 'pdi-trab10-theme';
  const toggle = document.getElementById('theme-toggle');
  const lLink = document.getElementById('theme-light');
  const dLink = document.getElementById('theme-dark');
  const label = document.getElementById('theme-label');

  function apply(mode) {
    const dark = mode === 'dark';
    document.body.classList.toggle('dark', dark);
    document.body.classList.toggle('light', !dark);
    if (lLink) lLink.disabled = dark;
    if (dLink) dLink.disabled = !dark;
    if (label) label.textContent = dark ? 'Dark' : 'Light';
    if (toggle) {
      toggle.setAttribute('aria-label', dark ? 'Alternar para tema claro' : 'Alternar para tema escuro');
      toggle.setAttribute('title', dark ? 'Tema claro (T)' : 'Tema escuro (T)');
    }
    try { localStorage.setItem(KEY, mode); } catch (e) { }
    window.dispatchEvent(new CustomEvent('pdi:themechange', { detail: { mode: mode } }));
  }

  function current() { return document.body.classList.contains('dark') ? 'dark' : 'light'; }

  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { saved = null; }
  apply(saved === 'dark' ? 'dark' : 'light');

  if (toggle) {
    toggle.addEventListener('click', function () {
      apply(current() === 'dark' ? 'light' : 'dark');
    });
  }

  window.PDITheme = { apply: apply, current: current };

  const PDI = { themeHooks: [], loops: [], disposers: [] };
  window.PDI = PDI;

  PDI.cssVar = function (name, fallback) {
    const v = getComputedStyle(document.body).getPropertyValue(name).trim();
    return v || (fallback == null ? '' : fallback);
  };
  PDI.format = function (v, digits) {
    const d = digits == null ? 2 : digits;
    if (!isFinite(v)) return '—';
    return v.toFixed(d);
  };
  PDI.formatPct = function (v, digits) {
    const d = digits == null ? 1 : digits;
    return (v * 100).toFixed(d) + '%';
  };
  PDI.clamp = function (v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); };
  PDI.interval = function (fn, ms) {
    let id = null;
    const h = {
      get running() { return id !== null; },
      start() { if (id === null) id = setInterval(fn, ms); return h; },
      stop() { if (id !== null) { clearInterval(id); id = null; } return h; },
      toggle() { if (id === null) h.start(); else h.stop(); return h.running; },
      restart() { h.stop(); h.start(); return h; }
    };
    PDI.loops.push(h);
    h.start();
    return h;
  };
  PDI.stopAll = function () { PDI.loops.forEach(function (h) { h.stop(); }); PDI.loops.length = 0; };
  PDI.onReset = function (fn) { PDI.disposers.push(fn); };
  PDI.resetAll = function () { PDI.stopAll(); PDI.disposers.forEach(function (f) { try { f(); } catch (e) { console.warn(e); } }); PDI.disposers.length = 0; };
  PDI.onTheme = function (fn) { if (PDI.themeHooks.indexOf(fn) === -1) PDI.themeHooks.push(fn); };

  const slides = Array.from(document.querySelectorAll('.slide'));
  const total = slides.length;
  let cur = -1;
  let hintShown = false;

  const el = {
    fill: document.getElementById('progress-fill'),
    counter: document.getElementById('slide-counter'),
    section: document.getElementById('section-indicator'),
    navToggle: document.getElementById('nav-toggle'),
    navItems: document.getElementById('nav-items'),
    fs: document.getElementById('fullscreen-btn'),
    hint: document.getElementById('keyboard-hint'),
    prev: document.getElementById('prev-btn'),
    next: document.getElementById('next-btn')
  };

  const navBtns = Array.from(el.navItems ? el.navItems.querySelectorAll('button[data-slide]') : []);

  function sectionOf(i) { return slides[i].getAttribute('data-section') || ''; }

  function update() {
    if (el.fill) el.fill.style.width = ((cur + 1) / total * 100) + '%';
    if (el.counter) el.counter.textContent = (cur + 1) + ' / ' + total;
    if (el.section) el.section.textContent = sectionOf(cur);
    navBtns.forEach(function (b) { b.classList.toggle('active', parseInt(b.dataset.slide, 10) === cur); });
    if (el.prev) el.prev.disabled = cur === 0;
    if (el.next) el.next.disabled = cur === total - 1;
  }

  function runInit(name) {
    String(name || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean).forEach(function (fn) {
      try {
        const parts = fn.split('.');
        const mod = window[parts[0]];
        if (parts.length > 1 && mod && typeof mod[parts[1]] === 'function') mod[parts[1]]();
        else if (typeof window[fn] === 'function') window[fn]();
        else console.warn('init não encontrado:', fn);
      } catch (e) { console.error('falha ao iniciar o slide:', fn, e); }
    });
  }

  function go(i, dir) {
    if (i < 0 || i >= total || i === cur) return;
    PDI.resetAll();
    slides.forEach(function (s) { s.classList.remove('active', 'from-right', 'from-left'); s.setAttribute('aria-hidden', 'true'); });
    cur = i;
    const s = slides[cur];
    s.classList.add('active');
    if (dir === 1) s.classList.add('from-right');
    if (dir === -1) s.classList.add('from-left');
    s.setAttribute('aria-hidden', 'false');
    s.querySelectorAll('.anim').forEach(function (n, k) {
      n.style.animation = 'none';
      void n.offsetWidth;
      n.style.animation = '';
      n.style.animationDelay = Math.min(k * 35, 350) + 'ms';
    });
    update();
    runInit(s.getAttribute('data-init'));
    showHint(false);
  }

  function next() { go(cur + 1, 1); }
  function prev() { go(cur - 1, -1); }

  function toggleFS() {
    if (!document.fullscreenElement) { const p = document.documentElement.requestFullscreen(); if (p && p.catch) p.catch(function () { }); }
    else if (document.exitFullscreen) { document.exitFullscreen(); }
  }
  function toggleMenu() {
    if (!el.navItems) return;
    const open = el.navItems.classList.toggle('open');
    if (el.navToggle) el.navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  function closeMenu() {
    if (!el.navItems) return;
    el.navItems.classList.remove('open');
    if (el.navToggle) el.navToggle.setAttribute('aria-expanded', 'false');
  }
  function showHint(v) { if (!el.hint) return; el.hint.classList.toggle('visible', !!v); }

  document.addEventListener('keydown', function (e) {
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'PageDown' || k === ' ') { e.preventDefault(); next(); }
    else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp') { e.preventDefault(); prev(); }
    else if (k === 'f' || k === 'F') { e.preventDefault(); toggleFS(); }
    else if (k === 't' || k === 'T') { e.preventDefault(); const btn = document.getElementById('theme-toggle'); if (btn) btn.click(); }
    else if (k === 'm' || k === 'M') { e.preventDefault(); toggleMenu(); }
    else if (k === 'Home') { e.preventDefault(); go(0, -1); }
    else if (k === 'End') { e.preventDefault(); go(total - 1, 1); }
    else if (k === 'Escape') { closeMenu(); }
  });

  if (el.navToggle) el.navToggle.addEventListener('click', toggleMenu);
  navBtns.forEach(function (b) { b.addEventListener('click', function () { go(parseInt(b.dataset.slide, 10), 1); closeMenu(); }); });
  if (el.fs) el.fs.addEventListener('click', toggleFS);
  if (el.prev) el.prev.addEventListener('click', prev);
  if (el.next) el.next.addEventListener('click', next);

  document.addEventListener('click', function (e) {
    const t = e.target && e.target.closest ? e.target.closest('[data-goto]') : null;
    if (!t) return;
    const i = parseInt(t.getAttribute('data-goto'), 10);
    if (!isNaN(i)) go(i, i > cur ? 1 : -1);
  });

  window.addEventListener('pdi:themechange', function () { PDI.themeHooks.forEach(function (fn) { try { fn(); } catch (e) { console.warn(e); } }); });

  let raf = null;
  window.addEventListener('resize', function () {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(function () { raf = null; PDI.themeHooks.forEach(function (fn) { try { fn(); } catch (e) { } }); });
  });

  PDI.go = go; PDI.next = next; PDI.prev = prev; PDI.total = total; PDI.current = function () { return cur; };

  function boot() {
    if (!total) { console.error('Nenhum slide encontrado.'); return; }
    go(0, 1);
    if (!hintShown) { hintShown = true; showHint(true); setTimeout(function () { showHint(false); }, 4200); }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
