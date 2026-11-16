/** Built-in V8 coverage, merging the three print suites against identical paths. */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const project = fileURLToPath(new URL('..', import.meta.url));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-print-coverage-'));
try {
  const files = ['bosco-print-context', 'bosco-print-ui', 'characteristic-print'];
  const result = spawnSync(process.execPath, [
    '--test', '--test-concurrency=1', '--experimental-test-coverage',
    ...files.map(name => `--test-coverage-include=**/bosco/js/${name}.js`),
    ...files.map(name => `apps/bosco/static/bosco/js/${name}.test.mjs`),
  ], { cwd: project, stdio: 'inherit', env: { ...process.env, ABIES_PRINT_TEST_STATIC_ROOT: root } });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
