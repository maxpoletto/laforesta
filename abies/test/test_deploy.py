"""Exercise release preparation with local Git repositories and a fake builder."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'bin' / 'deploy'
BUILD_STOP = 73


class DeployGitRemoteTests(unittest.TestCase):
    def test_fetches_release_tag_from_github_by_default(self):
        self.check_release_fetch('github')

    def test_fetches_release_tag_from_explicit_remote(self):
        self.check_release_fetch('upstream', remote_override='upstream')

    def check_release_fetch(self, remote_name, *, remote_override=None):
        with tempfile.TemporaryDirectory(prefix='abies-deploy-test-') as temp_dir:
            root = Path(temp_dir)
            upstream = root / 'upstream'
            checkout = root / 'checkout'
            upstream.mkdir()

            def git(repo, *args):
                return subprocess.run(
                    ['git', '-C', str(repo), *args],
                    check=True, capture_output=True, text=True,
                )

            git(upstream, 'init', '-b', 'main')
            git(upstream, 'config', 'user.name', 'Deploy Test')
            git(upstream, 'config', 'user.email', 'deploy@example.invalid')
            git(upstream, 'config', 'commit.gpgsign', 'false')
            app = upstream / 'abies'
            (app / 'bin').mkdir(parents=True)
            (app / 'compose').mkdir()
            (app / 'compose' / '.gitignore').write_text('.env.*\n')
            branding = app / 'bin' / 'sync-branding'
            branding.write_text('#!/usr/bin/env bash\nexit 0\n')
            branding.chmod(0o755)
            marker = app / 'release.txt'
            marker.write_text('original checkout')
            git(upstream, 'add', '.')
            git(upstream, 'commit', '-m', 'Initial application')
            git(upstream, 'clone', '--no-tags', '--origin', remote_name,
                str(upstream), str(checkout))

            # Publish a release that the checkout has not fetched yet.
            marker.write_text('fetched release')
            git(upstream, 'commit', '-am', 'Release application')
            git(upstream, '-c', 'tag.gpgsign=false', 'tag', '-a',
                'abies-test-release', '-m', 'Test release')
            local_app = checkout / 'abies'
            (local_app / 'compose' / '.env.prod').write_text('ABIES_INSTANCE=prod\n')

            # Stop at the first build command, before any deployment operation.
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            builder = fake_bin / 'docker-compose'
            builder.write_text(
                '#!/usr/bin/env bash\n'
                f'exec {shlex.quote(sys.executable)} -c '
                + shlex.quote(
                    'import json, os, pathlib, sys; '
                    'print(json.dumps({'
                    '"args": sys.argv[1:], '
                    '"version": os.environ.get("ABIES_VERSION"), '
                    '"source": pathlib.Path("release.txt").read_text()})); '
                    f'sys.exit({BUILD_STOP})'
                ) + ' "$@"\n'
            )
            builder.chmod(0o755)
            env = dict(os.environ, PATH=f'{fake_bin}{os.pathsep}{os.environ["PATH"]}')
            env.pop('ABIES_GIT_REMOTE', None)
            if remote_override is not None:
                env['ABIES_GIT_REMOTE'] = remote_override

            result = subprocess.run(
                ['bash', str(SCRIPT), 'prod', 'abies-test-release'],
                cwd=local_app, env=env, capture_output=True, text=True,
            )

            self.assertEqual(result.returncode, BUILD_STOP, result.stderr)
            build = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(build['args'][-1], 'build')
            self.assertEqual(build['version'], 'abies-test-release')
            self.assertEqual(build['source'], 'fetched release')
            self.assertNotIn('==> pre-deploy backup', result.stdout)
