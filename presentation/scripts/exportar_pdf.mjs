// PDF de contingência (uma página por slide), offline.
// Uso (em presentation/):  npm install && npm run pdf
import { chromium } from "playwright-core";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import { chromePath } from "./navegador.mjs";

const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(dir, "seminario-trab10.pdf");
const browser = await chromium.launch({ executablePath: chromePath(), args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.route("**/*", (r) => (r.request().url().startsWith("file://") ? r.continue() : r.abort()));
await page.goto(pathToFileURL(path.join(dir, "index.html")).href + "?print-pdf");
await page.waitForFunction(() => document.querySelectorAll(".pdf-page").length > 10 && document.fonts.status === "loaded");
await page.waitForTimeout(2000);
await page.pdf({ path: out, preferCSSPageSize: true, printBackground: true });
await browser.close();
console.log("PDF gerado:", path.relative(process.cwd(), out));
