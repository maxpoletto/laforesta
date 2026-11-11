"""The image recipe must refresh packages inherited from its pinned base."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


DOCKERFILE = Path(__file__).resolve().parents[1] / 'Dockerfile'

FAKE_APT = '''
import json
import os
import sys
from pathlib import Path

path = Path(os.environ['OS_LAYER_STATE'])
state = json.loads(path.read_text())
args = sys.argv[1:]
if args[0] == 'update':
    state['fresh_index'] = True
elif args[0] == 'upgrade':
    assert state['fresh_index'], 'Refresh package indexes before upgrading'
    if os.environ.get('FAIL_OS_UPGRADE'):
        sys.exit(23)
    state['installed'] = dict.fromkeys(state['installed'], 'patched')
elif args[0] == 'install':
    assert state['fresh_index'], 'Refresh package indexes before installing'
    for name in args[1:]:
        if not name.startswith('-'):
            state['installed'][name] = 'patched'
else:
    sys.exit('Unexpected apt-get invocation: ' + repr(args))
path.write_text(json.dumps(state))
'''

FAKE_CLEANUP = '''
import json
import os
from pathlib import Path

path = Path(os.environ['OS_LAYER_STATE'])
state = json.loads(path.read_text())
state['cleaned'] = True
path.write_text(json.dumps(state))
'''


class ContainerOsUpdateTests(unittest.TestCase):
    def test_os_layer_upgrades_inherited_packages(self):
        result, state = self.run_os_layer()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state['installed']['base-library'], 'patched',
                         'Installing new dependencies must also patch the base image')
        self.assertIn('libexpat1', state['installed'])
        self.assertIn('terser', state['installed'])
        self.assertTrue(state['cleaned'])

    def test_failed_os_upgrade_aborts_the_layer(self):
        result, state = self.run_os_layer(fail_upgrade=True)

        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertNotIn('terser', state['installed'])
        self.assertFalse(state['cleaned'])

    def run_os_layer(self, *, fail_upgrade=False):
        lines = iter(DOCKERFILE.read_text().splitlines())
        command = next(line[4:] for line in lines if line.startswith('RUN '))
        while command.endswith('\\'):
            command = command[:-1] + next(lines)

        with tempfile.TemporaryDirectory(prefix='abies-os-layer-test-') as directory:
            root = Path(directory)
            state_path = root / 'state.json'
            state_path.write_text(json.dumps({
                'installed': {'base-library': 'unpatched'},
                'fresh_index': False,
                'cleaned': False,
            }))
            for name, source in (('apt-get', FAKE_APT), ('rm', FAKE_CLEANUP)):
                executable = root / name
                executable.write_text(f'#!{sys.executable}\n' + textwrap.dedent(source))
                executable.chmod(0o755)
            env = dict(os.environ)
            env['PATH'] = f'{root}{os.pathsep}{env["PATH"]}'
            env['OS_LAYER_STATE'] = str(state_path)
            env.pop('FAIL_OS_UPGRADE', None)
            if fail_upgrade:
                env['FAIL_OS_UPGRADE'] = '1'
            result = subprocess.run(
                ['sh', '-c', command], env=env, capture_output=True, text=True,
            )
            return result, json.loads(state_path.read_text())
