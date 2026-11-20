// Renders REAL captured text (never typed in by hand) as images, for the evidence items that are not browser screens:
//   10  offline server console   <- docs/screenshots/evidence/offline_server_console.txt (real server output, trimmed and labelled)
//   11  automated test result    <- docs/screenshots/evidence/pytest_run.txt (real pytest output)
//   12  project structure        <- the actual folder tree, read from the file system when this script runs
//   09b CSV contents             <- docs/screenshots/09_downloaded_ranking_combined.csv (the file the app produced)
// These are rendered pictures of genuine output, NOT screenshots of a terminal window or an editor; every image says so in its footer.
// Tooling only. Run:  node presentation/tools/render_text_images.js   (NODE_PATH must contain playwright, see render_images.js)
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..", "..");
const shots = path.join(root, "docs", "screenshots");
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const terminal = (title, body, note) => `<html><body style="margin:0;background:#fff;font-family:Segoe UI,Arial,sans-serif">
<div style="margin:24px;border-radius:10px;overflow:hidden;border:1px solid #2b3245;width:1100px">
<div style="background:#2b3245;color:#cfd6e6;padding:8px 14px;font-size:13px">${esc(title)}</div>
<pre style="margin:0;background:#12161f;color:#e6e9f0;padding:16px 18px;font:14px/1.45 Consolas,'Courier New',monospace;white-space:pre-wrap;word-break:break-all">${esc(body)}</pre>
<div style="background:#eef1f6;color:#5b6475;padding:8px 14px;font-size:12px">${esc(note)}</div></div></body></html>`;

function tree(dir, prefix = "", depth = 0) {
  const skip = new Set([".venv", "node_modules", "__pycache__", ".pytest_cache"]);
  let lines = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true }).filter((e) => !skip.has(e.name))
    .sort((a, b) => (b.isDirectory() - a.isDirectory()) || a.name.localeCompare(b.name));
  entries.forEach((e, i) => {
    const last = i === entries.length - 1;
    lines.push(prefix + (last ? "`-- " : "|-- ") + e.name + (e.isDirectory() ? "/" : ""));
    if (e.isDirectory() && depth < 2) lines = lines.concat(tree(path.join(dir, e.name), prefix + (last ? "    " : "|   "), depth + 1));
  });
  return lines;
}

function csvRows(text) {
  const rows = []; let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; }
    else if (c === '"') q = true; else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; } else if (c !== "\r") cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

(async () => {
  const browser = await chromium.launch({ channel: "msedge" });
  const page = await (await browser.newContext({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1.5 })).newPage();
  const shoot = async (html, name) => {
    await page.setContent(html);
    await page.locator("body > div").first().screenshot({ path: path.join(shots, name) });
    console.log("saved", name);
  };
  const note = "Rendered picture of genuine captured text (file named above); not a screenshot of a terminal window.";

  const consoleText = fs.readFileSync(path.join(shots, "evidence", "offline_server_console.txt"), "utf8");
  await shoot(terminal("Offline run: server console (docs/screenshots/evidence/offline_server_console.txt)", consoleText, note), "10_offline_model_execution_console_rendered.png");

  const testText = fs.readFileSync(path.join(shots, "evidence", "pytest_run.txt"), "utf8");
  await shoot(terminal("Automated tests (docs/screenshots/evidence/pytest_run.txt)", testText, note), "11_automated_tests_389_passed_rendered.png");

  const treeText = "reusme_matcher/\n" + tree(root).join("\n");
  await shoot(terminal("Project structure (read from the file system when the image was made; .venv, node_modules and caches hidden)", treeText,
    "Rendered picture of the real folder listing; not a screenshot of an editor."), "12_project_structure_rendered.png");

  const rows = csvRows(fs.readFileSync(path.join(shots, "09_downloaded_ranking_combined.csv"), "utf8")).filter((r) => r.length > 1);
  const head = rows[0];
  const lastCol = rows[0].length - 1;
  const table = `<html><body style="margin:0;background:#fff;font-family:Segoe UI,Arial,sans-serif"><div style="margin:24px;width:2450px">
<div style="font-size:15px;font-weight:600;margin-bottom:8px">ranking_combined.csv, exactly as downloaded from the app (synthetic demo data)</div>
<table style="border-collapse:collapse;font-size:12.5px;width:100%"><tr>${head.map((h) => `<th style="text-align:left;background:#eef1f6;border:1px solid #cfd5e1;padding:5px 7px">${esc(h)}</th>`).join("")}</tr>
${rows.slice(1).map((r) => `<tr>${r.map((c, ci) => `<td style="border:1px solid #dfe3ec;padding:4px 7px;min-width:60px;${ci === lastCol ? "min-width:560px;" : ""}${c === "" ? "background:#fafafa" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</table>
<div style="color:#5b6475;font-size:12px;margin-top:8px">Rendered picture of the downloaded file docs/screenshots/09_downloaded_ranking_combined.csv. Empty cells are really empty. Excluded files have no rank and no scores.</div></div></body></html>`;
  await page.setViewportSize({ width: 2550, height: 700 });
  await shoot(table, "09b_csv_contents_rendered.png");
  await browser.close();
})();
