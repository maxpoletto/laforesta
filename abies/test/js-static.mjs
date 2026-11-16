/** Stage the vendored numerical dependency beside copied browser modules. */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

export function copyGeodesicVendor(staticRoot) {
  fs.cpSync(fileURLToPath(new URL('../apps/base/static/vendor/geographiclib', import.meta.url)),
    path.join(staticRoot, 'vendor', 'geographiclib'), { recursive: true });
}

/** Use one source path across isolated tests when measuring merged coverage. */
export function stagePrintModules(prefix) {
  const shared = process.env.ABIES_PRINT_TEST_STATIC_ROOT;
  const root = shared || fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  for (const app of ['base', 'bosco']) {
    fs.cpSync(fileURLToPath(new URL(`../apps/${app}/static/${app}/js`, import.meta.url)),
      path.join(root, app, 'js'), { recursive: true });
  }
  copyGeodesicVendor(root);
  if (!shared) process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}
