"""Fail-closed macOS research-worker sandbox. Trusted owner setup is outside it."""
import json
import os
from pathlib import Path
import subprocess
import sys


def command(arguments, root):
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise ValueError('Sealed campaigns currently require the tested macOS sandbox-exec backend')
    source = Path(__file__).resolve().parent.parent
    # No home-directory allowance, Docker socket, network, or process inspection.
    reads = {str(source), str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()),
             str(Path(root).resolve()), '/System', '/usr', '/bin', '/sbin', '/Library',
             '/opt/homebrew', '/private/etc', '/private/var/db/dyld', '/dev'}
    # Literal / matches only the root directory, not descendants. Regression tests
    # prove outside-file reads and network access remain denied.
    rules = ['(version 1)', '(deny default)', '(allow process-exec process-fork)',
             '(allow sysctl-read)', '(allow mach-lookup)', '(allow signal (target self))',
             '(allow file-read-metadata)', '(allow file-read* (literal "/"))', '(allow file-write* (literal "/dev/null"))']
    rules += [f'(allow file-read* (subpath {json.dumps(path)}))' for path in sorted(reads)]
    rules += [f'(allow file-write* (subpath {json.dumps(str(Path(root).resolve()))}))']
    return ['/usr/bin/sandbox-exec', '-p', '\n'.join(rules), *arguments]


def probe(root, forbidden_path):
    """Exit zero only when an actual OS read is denied (not a missing-file check)."""
    code = ('from pathlib import Path; import sys\n'
            'try: Path(sys.argv[1]).read_bytes()\n'
            'except PermissionError: sys.exit(0)\n'
            'except Exception: sys.exit(2)\n'
            'sys.exit(1)\n')
    return subprocess.run(command([sys.executable, '-c', code, str(forbidden_path)], root),
                          capture_output=True, timeout=10).returncode == 0


def worker_environment(root):
    temporary = Path(root).resolve() / 'worker-tmp'
    temporary.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
    env.update(TMPDIR=str(temporary), TORCHINDUCTOR_CACHE_DIR=str(temporary / 'torch'))
    return env
