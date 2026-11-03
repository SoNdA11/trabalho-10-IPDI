// Miniaturas dos slides para os guias: docs/img/slide-XX.jpg (todas as animações concluídas).
// Uso (em presentation/):  npm install && npm run miniaturas   [TEMA=escuro]
import { chromium } from "playwright-core";
import { fileURLToPath, pathToFileURL } from "node:url";
import fs from "node:fs";
import path from "node:path";
import { chromePath } from "./navegador.mjs";

const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.resolve(dir, "..", "docs", "img");
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ executablePath: chromePath(), args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
page.on("console", (m) => { if (m.type() === "error") console.error("console:", m.text()); });
page.on("pageerror", (e) => console.error("pageerror:", e.message));
await page.route("**/*", (r) => (r.request().url().startsWith("file://") ? r.continue() : r.abort()));
await page.goto(pathToFileURL(path.join(dir, "index.html")).href + `?tema=${process.env.TEMA || "claro"}`);
await page.waitForFunction(() => window.Reveal && Reveal.isReady() && document.fonts.status === "loaded");
const total = await page.evaluate(() => Reveal.getTotalSlides());
const only = process.env.SLIDES ? process.env.SLIDES.split(",").map(Number) : null;
for (let h = 0; h < total; h++) {
  if (only && !only.includes(h + 1)) continue;
  await page.evaluate((h) => { Reveal.slide(h, 0); while (Reveal.nextFragment()); }, h);
  await page.waitForTimeout(2600);
  const nome = `slide-${String(h + 1).padStart(2, "0")}.jpg`;
  await page.screenshot({ path: path.join(out, nome), type: "jpeg", quality: 82 });
  console.log(nome);
}
await browser.close();
