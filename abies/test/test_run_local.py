"""The local launcher shuts down both children when either exits."""
import errno
import runpy
import signal
import sys
from pathlib import Path
from unittest.mock import Mock, call


def launcher_module():
    launcher = Path(__file__).resolve().parents[1] / 'bin' / 'run-local.py'
    return runpy.run_path(str(launcher))


def test_exited_server_group_does_not_interrupt_worker_cleanup(monkeypatch):
    modules = launcher_module()
    monkeypatch.setitem(modules['main'].__globals__, 'replace_server', Mock())
    server = Mock(pid=101)
    server.poll.return_value = 1
    worker = Mock(pid=102)
    spawn = Mock(side_effect=[server, worker])
    kill_group = Mock(side_effect=[PermissionError(errno.EPERM, 'Operation not permitted'), None])
    monkeypatch.setattr(modules['subprocess'], 'Popen', spawn)
    monkeypatch.setattr(modules['signal'], 'signal', Mock())
    monkeypatch.setattr(modules['os'], 'killpg', kill_group)

    assert modules['main'](['8001']) == 1
    assert spawn.call_args_list == [
        call([sys.executable, 'manage.py', 'runserver', '8001'], start_new_session=True),
        call([sys.executable, 'manage.py', 'run_terrain_worker'], start_new_session=True),
    ]
    assert kill_group.call_args_list == [call(server.pid, signal.SIGTERM), call(worker.pid, signal.SIGTERM)]
    server.wait.assert_called_once()
    worker.wait.assert_called_once()


def test_replacing_local_stack_signals_its_launcher(monkeypatch, tmp_path):
    modules = launcher_module()
    inspect = Mock(side_effect=[
        '101\n',
        '100 S 1000 /tmp/env/bin/python3 manage.py runserver 8001\n', f'p101\nfcwd\nn{tmp_path}\n',
        '99 S 1000 /tmp/env/bin/python3 manage.py runserver 8001\n', f'p100\nfcwd\nn{tmp_path}\n',
        '89 S 1000 /tmp/env/bin/python3 bin/run-local.py 8001\n', f'p99\nfcwd\nn{tmp_path}\n',
        '89 S 1000 /tmp/env/bin/python3 bin/run-local.py 8001\n', f'p99\nfcwd\nn{tmp_path}\n',
        '',
    ])
    stop = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(modules['replace_server'].__globals__, 'command_output', inspect)
    monkeypatch.setattr(modules['os'], 'getuid', lambda: 1000)
    monkeypatch.setattr(modules['os'], 'kill', stop)
    modules['replace_server'](8001)
    stop.assert_called_once_with(99, signal.SIGTERM)


def test_listener_in_another_checkout_is_left_running(monkeypatch, tmp_path):
    modules = launcher_module()
    inspect = Mock(side_effect=[
        '101\n', '100 S 1000 /tmp/env/bin/python3 manage.py runserver 8001\n',
        f'p101\nfcwd\nn{tmp_path / "other"}\n',
    ])
    stop = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(modules['replace_server'].__globals__, 'command_output', inspect)
    monkeypatch.setattr(modules['os'], 'getuid', lambda: 1000)
    monkeypatch.setattr(modules['os'], 'kill', stop)
    modules['replace_server'](8001)
    stop.assert_not_called()
