// Renders the SVG diagrams and charts in docs/figures/ to PNG files in presentation/images/ with the installed Microsoft Edge.
// Tooling only (not part of the application; not in requirements.txt). Setup, outside the project folder:
//   npm install playwright pptxgenjs        (Playwright drives the Edge that is already installed; no browser download)
// Run from the project folder:   set NODE_PATH=<folder with node_modules>   then   node presentation/tools/render_images.js
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..", "..");
const figures = path.join(root, "docs", "figures");
const outDir = path.join(root, "presentation", "images");

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge" });
  const files = fs.readdirSync(figures).filter((f) => f.endsWith(".svg"));
  for (const file of files) {
    const svg = fs.readFileSync(path.join(figures, file), "utf8");
    const m = svg.match(/viewBox="0 0 (\d+) (\d+)"/);
    const [w, h] = [parseInt(m[1], 10), parseInt(m[2], 10)];
    const context = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, colorScheme: "light" });
    const page = await context.newPage();
    await page.goto("file:///" + path.join(figures, file).replace(/\\/g, "/"));
    const target = path.join(outDir, file.replace(/\.svg$/, ".png"));
    await page.screenshot({ path: target, clip: { x: 0, y: 0, width: w, height: h } });
    console.log("rendered", path.relative(root, target), w + "x" + h);
    await context.close();
  }
  await browser.close();
})();
