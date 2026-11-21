"""The local launcher shuts down both children when either exits."""
import errno
import runpy
import signal
import sys
from pathlib import Path
from unittest.mock import Mock, call


def test_exited_server_group_does_not_interrupt_worker_cleanup(monkeypatch):
    launcher = Path(__file__).resolve().parents[1] / 'bin' / 'run-local.py'
    modules = runpy.run_path(str(launcher))
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
