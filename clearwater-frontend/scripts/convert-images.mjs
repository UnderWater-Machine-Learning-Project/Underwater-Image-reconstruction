/**
 * One-off script: Convert all campaign PNG images to WebP.
 * Usage: node scripts/convert-images.mjs
 *
 * - Outputs .webp files alongside the original PNGs.
 * - Quality 82 is a good balance between size and visual fidelity.
 * - Original PNGs are preserved — delete manually when ready.
 */

import { readdir, stat } from 'node:fs/promises';
import { join, extname, basename } from 'node:path';
import sharp from 'sharp';

const CAMPAIGN_DIR = join(process.cwd(), 'public', 'campaign', 'images');
const WEBP_QUALITY = 82;

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(full)));
    } else if (extname(entry.name).toLowerCase() === '.png') {
      files.push(full);
    }
  }
  return files;
}

async function convert() {
  const pngs = await walk(CAMPAIGN_DIR);
  console.log(`Found ${pngs.length} PNG files to convert.\n`);

  for (const png of pngs) {
    const webp = png.replace(/\.png$/i, '.webp');
    const name = basename(png);

    const before = (await stat(png)).size;
    await sharp(png).webp({ quality: WEBP_QUALITY }).toFile(webp);
    const after = (await stat(webp)).size;

    const saved = ((1 - after / before) * 100).toFixed(1);
    console.log(
      `  ${name} → ${basename(webp)}  ` +
      `${(before / 1024 / 1024).toFixed(2)} MB → ${(after / 1024 / 1024).toFixed(2)} MB  ` +
      `(${saved}% smaller)`
    );
  }

  console.log('\nDone. Update manifest.ts paths from .png to .webp.');
}

convert().catch((err) => {
  console.error('Conversion failed:', err);
  process.exit(1);
});
