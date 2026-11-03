// Captura telas da interface (src/app.py precisa estar rodando em localhost:5000).
// Uso (em presentation/): node scripts/capturar_demo.mjs
import { chromium } from "playwright-core";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromePath } from "./navegador.mjs";

const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const url = process.env.URL || "http://localhost:5000";
const browser = await chromium.launch({ executablePath: chromePath(), args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("pageerror:", e.message));
await page.addInitScript(() => localStorage.setItem("trab10-tema", "claro"));
await page.goto(url);
await page.waitForSelector("#samples button");
await page.selectOption("#classFilter", "River");
await page.click("#samples button:nth-child(3)");
await page.waitForSelector("#chipResult .badge");
await page.waitForTimeout(600);
await page.screenshot({ path: path.join(dir, "img", "demo_imagem.png") });
await page.click('.tabs button[data-tab="scene"]');
await page.waitForSelector("#wMap[src^='data:']", { timeout: 60000 });
const box = await page.locator("#sceneImg").boundingBox();
await page.mouse.click(box.x + box.width * 0.47, box.y + box.height * 0.5);
await page.waitForTimeout(2500);
await page.screenshot({ path: path.join(dir, "img", "demo_interface.png") });
await page.click('.tabs button[data-tab="metrics"]');
await page.waitForSelector(".mcard");
await page.waitForTimeout(500);
await page.screenshot({ path: path.join(dir, "img", "demo_metricas.png") });
await browser.close();
console.log("capturas em presentation/img/demo_*.png");
