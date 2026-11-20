// Captures genuine screenshots of the RUNNING app (http://localhost:8601 by default) into docs/screenshots/ using the installed Edge.
// Tooling only (not part of the application). It uploads the SYNTHETIC files in data/sample_resumes/ and uses the app's synthetic demo job description.
// Start the app first, for example:   $env:HF_HUB_OFFLINE = "1"; streamlit run app.py --server.port 8601
// Run (Playwright installed outside the project, see render_images.js):   node presentation/tools/capture_screenshots.js
// Environment: APP_URL (default http://localhost:8601), COVERAGE_X (x offset of the coverage column header inside the results table).
// Only the page content is captured (no browser window, no desktop), so no personal information from the machine can appear.
// The Streamlit toolbar ("Deploy" menu) is hidden by a style tag because it is not part of this project.
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..", "..");
const outDir = path.join(root, "docs", "screenshots");
const url = process.env.APP_URL || "http://localhost:8601";
const coverageX = parseInt(process.env.COVERAGE_X || "1010", 10);
const demo = path.join(root, "data", "sample_resumes", "backend_demo");
const edge = path.join(root, "data", "sample_resumes", "edge_cases");
const list = (dir) => fs.readdirSync(dir).filter((f) => f.endsWith(".pdf")).sort().map((f) => path.join(dir, f));
const demoFiles = list(demo);
const edgeFiles = list(edge);

fs.mkdirSync(outDir, { recursive: true });
const W = 1720, H = 3400;

async function clipShot(page, name, topY, bottomY) {
  const y = Math.max(0, topY);
  const height = Math.min(bottomY - y, H - y);
  await page.screenshot({ path: path.join(outDir, name), clip: { x: 0, y, width: W, height } });
  console.log("saved", name, W + "x" + Math.round(height));
}

async function waitIdle(page, ms = 1500) {
  await page.waitForTimeout(ms);
  await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 600000 }).catch(() => {});
  await page.waitForTimeout(600);
}

(async () => {
  const browser = await chromium.launch({ channel: "msedge" });
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, colorScheme: "light", acceptDownloads: true });
  const page = await context.newPage();
  await page.goto(url);
  await page.addStyleTag({ content: '[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }' });
  const h = (n) => page.getByRole("heading", { name: n, exact: true });
  await h("1. Job description").waitFor({ timeout: 120000 });
  await waitIdle(page, 2500);
  const top = async (n) => (await h(n).boundingBox()).y;

  // 01 home screen
  await page.screenshot({ path: path.join(outDir, "01_home_screen.png"), clip: { x: 0, y: 0, width: W, height: 940 } });
  console.log("saved 01_home_screen.png");

  // 02 job description + skills
  await page.getByRole("button", { name: "Fill in this job description" }).click();
  await waitIdle(page);
  await clipShot(page, "02_job_description_and_skills.png", (await top("1. Job description")) - 20, (await top("2. Resumes")) - 10);

  // 03 nine demo resumes
  await page.setInputFiles('input[type="file"]', demoFiles);
  await page.getByText("synthetic_resume_09.pdf").first().waitFor({ state: "attached", timeout: 60000 });
  await waitIdle(page, 3000);
  await clipShot(page, "03_multiple_resume_uploads.png", (await top("2. Resumes")) - 20, (await top("3. Rank")) - 10);

  // 04 add the six edge-case files (the uploader appends to what is already there)
  await page.setInputFiles('input[type="file"]', edgeFiles);
  await page.getByText("long_resume_37_pages.pdf").first().waitFor({ state: "attached", timeout: 60000 });
  await waitIdle(page, 4000);
  await clipShot(page, "04_file_validation_and_edge_cases.png", (await top("2. Resumes")) - 20, (await top("3. Rank")) - 10);
  const statusTable = page.locator('[data-testid="stDataFrame"]').first();
  const sBox = await statusTable.boundingBox();
  await page.mouse.move(sBox.x + sBox.width / 2, sBox.y + sBox.height / 2);
  await page.mouse.wheel(0, 700);
  await page.waitForTimeout(1200);
  await clipShot(page, "04b_file_validation_edge_cases_scrolled.png", (await top("2. Resumes")) - 20, (await top("3. Rank")) - 10);

  const rank = async (label) => {
    await page.getByRole("button", { name: "Rank candidates" }).click();
    await page.getByRole("heading", { name: `Method: ${label}`, exact: true }).waitFor({ timeout: 600000 });
    await waitIdle(page, 3000);
  };
  const resultsClip = async (name) => clipShot(page, name, (await top("4. Results")) - 2, (await top("Candidate details")) - 8);

  // combined ranking, sorted by coverage
  await rank("Combined ranking (default)");
  console.log("combined ranking done");
  // 08 candidate details (first expander of the combined ranking, table still in index order)
  await page.mouse.move(5, 5);
  const firstDetails = page.locator("details").filter({ hasText: "#1 " }).first();
  await firstDetails.locator("summary").click();
  await page.waitForTimeout(1500);
  const dBox = await firstDetails.boundingBox();
  await clipShot(page, "08_candidate_details_and_score_explanation.png", (await top("4. Results")) + 60, dBox.y + dBox.height + 24);

  const df = page.locator('[data-testid="stDataFrame"]').nth(1);
  const box = await df.boundingBox();
  await page.mouse.click(box.x + coverageX, box.y + 17);
  await page.waitForTimeout(800);
  await page.mouse.click(box.x + coverageX, box.y + 17);   // second click: descending
  await page.waitForTimeout(1200);
  await resultsClip("05_combined_ranking_sorted_by_coverage.png");

  // 09 download the CSV of the combined run
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download ranking as CSV" }).click()]);
  await download.saveAs(path.join(outDir, "09_downloaded_ranking_combined.csv"));
  const dl = await page.getByRole("button", { name: "Download ranking as CSV" }).boundingBox();
  await page.screenshot({ path: path.join(outDir, "09a_csv_download_button.png"), clip: { x: 0, y: dl.y - 260, width: W, height: 330 } });
  console.log("saved 09a_csv_download_button.png and the CSV");

  // TF-IDF and semantic
  await page.getByText("TF-IDF baseline", { exact: true }).first().click();
  await waitIdle(page);
  await rank("TF-IDF baseline");
  await resultsClip("06_tfidf_ranking.png");
  await page.getByText("Semantic matching", { exact: true }).first().click();
  await waitIdle(page);
  await rank("Semantic matching");
  await resultsClip("07_semantic_ranking.png");

  await browser.close();
})().catch((e) => { console.error("FAILED:", e.message); process.exit(1); });
