/** Stage the vendored numerical dependency beside copied browser modules. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function copyGeodesicVendor(staticRoot) {
  fs.cpSync(fileURLToPath(new URL('../apps/base/static/vendor/geographiclib', import.meta.url)),
    path.join(staticRoot, 'vendor', 'geographiclib'), { recursive: true });
}
