import argparse
import json
from pathlib import Path
import sys

from .data import Config, generate, profile, validate
from .engine import create, execute, report
from .io import read


def main(argv=None):
    parser = argparse.ArgumentParser(prog='graphsmith', description='Auditable graph-learning development experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'profile', 'research'):
        command = sub.add_parser(name)
        command.add_argument('config', type=Path)
        if name == 'research':
            command.add_argument('--output', type=Path, default=Path('runs'))
            command.add_argument('--evaluator', help='Private local evaluator URL (macOS sealed campaigns)')
    for name in ('resume', 'report', 'freeze', 'finalize'):
        sub.add_parser(name).add_argument('run', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {'freeze', 'finalize'}:
            from .finalization import freeze, finalize
            print((freeze if args.command == 'freeze' else finalize)(args.run))
        elif args.command == 'resume':
            print(execute(args.run))
        elif args.command == 'report':
            print(report(args.run))
        else:
            config = Config.parse(read(args.config))
            if args.command == 'research':
                root = create(config, args.output, args.evaluator)
                print(f'Campaign: {root}', flush=True)
                print(execute(root))
            else:
                bundle = generate(config)
                print(json.dumps(profile(bundle) if args.command == 'profile' else validate(bundle), indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'graphsmith: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
