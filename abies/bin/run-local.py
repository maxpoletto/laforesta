"""Run the local Django server and terrain worker together."""
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def command_output(*args):
    try:
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    command: str

    def runs(self, script, action=''):
        for path in (script, str(Path.cwd() / script)):
            entry = f'{path} {action}' if action else path
            interpreter, match, options = self.command.partition(f' {entry}')
            if match and (not options or options.startswith(' ')) and re.fullmatch(
                r'python[\d.]*', Path(interpreter).name.lower(),
            ):
                return True
        return False


def checkout_process(pid):
    fields = command_output('ps', '-ww', '-p', str(pid), '-o', 'ppid=', '-o', 'stat=',
                            '-o', 'uid=', '-o', 'command=').split(maxsplit=3)
    if len(fields) != 4:
        return None
    parent, state, uid, command = fields
    if int(uid) != os.getuid() or state.startswith('Z'):
        return None
    cwd = command_output('lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn').splitlines()
    if any(line.startswith('n') and Path(line[1:]).resolve() == Path.cwd() for line in cwd):
        return Process(pid, int(parent), command.strip())
    return None


def replace_server(port):
    listeners = set(command_output('lsof', '-a', '-t', f'-iTCP:{port}', '-sTCP:LISTEN').split())
    if len(listeners) != 1:
        return
    pid = listeners.pop()
    server = checkout_process(int(pid)) if pid.isdecimal() else None
    if not server or not server.runs('manage.py', 'runserver'):
        return
    parent = checkout_process(server.parent)
    reloader = parent if parent and parent.runs('manage.py', 'runserver') else None
    launcher = checkout_process(reloader.parent) if reloader else parent
    if launcher and launcher.runs('bin/run-local.py'):
        targets = [launcher]
    else:
        targets = [reloader, server] if reloader else [server]

    print(f'Stopping the Abies server from this checkout on port {port}.', flush=True)
    for process in targets:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 8
    remaining = targets
    while remaining:
        remaining = [p for p in remaining if checkout_process(p.pid) == p]
        if remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for Abies on port {port} to stop.')
            time.sleep(0.1)


def main(server_args=()):
    port = server_args[0] if server_args else '8000'
    if not port.isascii() or not port.isdecimal() or not 0 < int(port) <= 65535:
        print('LOCAL_PORT must be a TCP port number.', file=sys.stderr)
        return 2
    try:
        replace_server(int(port))
    except TimeoutError as exc:
        print(exc, file=sys.stderr)
        return 1
    stop_signal = None

    def stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    processes = []
    try:
        for command in [('runserver', *server_args), ('run_terrain_worker',)]:
            processes.append(subprocess.Popen(
                [sys.executable, 'manage.py', *command], start_new_session=True,
            ))
        while stop_signal is None:
            for process in processes:
                code = process.poll()
                if code is not None:
                    return 128 - code if code < 0 else code
            time.sleep(0.2)
        return 128 + stop_signal
    finally:
        for process in processes:
            # The Django reloader also starts a child in the process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                # An exited process group can also report EPERM.
                if process.poll() is None:
                    process.terminate()
        for process in processes:
            process.wait()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
