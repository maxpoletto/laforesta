"""Run the local Django server and terrain worker with a shared lifetime."""
import os
import signal
import subprocess
import sys
import time


def main(server_args=()):
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
