const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/addld/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const paperDir = __dirname;
  const htmlPath = path.join(paperDir, 'Multi_Model_Speaker_Recognition_IEEE_Paper.html');
  const pdfPath = path.join(paperDir, 'Multi_Model_Speaker_Recognition_IEEE_Paper.pdf');
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe'
  });
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'networkidle' });
    await page.emulateMedia({ media: 'print' });
    await page.pdf({
      path: pdfPath,
      format: 'Letter',
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: true,
      headerTemplate: '<div></div>',
      footerTemplate: '<div style="width:100%;text-align:center;font-family:Times New Roman;font-size:8px;color:#555;"><span class="pageNumber"></span></div>',
      margin: { top: '0', right: '0', bottom: '0', left: '0' }
    });
    console.log(pdfPath);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
