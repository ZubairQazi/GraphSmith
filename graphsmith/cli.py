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
    from .dpsyn import add_parser
    add_parser(sub)
    from .dpsyn_campaign import add_parser as add_campaign_parser
    add_campaign_parser(sub)
    from .dpsyn_diagnose import add_parser as add_diagnosis_parser
    add_diagnosis_parser(sub)
    from .dpsyn_selectivity import add_parser as add_selectivity_parser
    add_selectivity_parser(sub)
    from .dpsyn_channels import add_parser as add_channels_parser
    add_channels_parser(sub)
    from .dpsyn_review import add_parser as add_review_parser
    add_review_parser(sub)
    from .dpsyn_feature_audit import add_parser as add_feature_audit_parser
    add_feature_audit_parser(sub)
    from .dpsyn_research import add_parser as add_research_parser
    add_research_parser(sub)
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
        if args.command == 'dpsyn-research':
            from .dpsyn_research import run
            print(json.dumps(run(args), indent=2))
        elif args.command == 'dpsyn-feature-audit':
            from .dpsyn_feature_audit import run
            print(run(args))
        elif args.command == 'dpsyn-review':
            from .dpsyn_review import run
            print(run(args))
        elif args.command == 'dpsyn-channels':
            from .dpsyn_channels import run
            print(run(args))
        elif args.command == 'dpsyn-selectivity':
            from .dpsyn_selectivity import run
            print(run(args))
        elif args.command == 'dpsyn-diagnose':
            from .dpsyn_diagnose import run
            print(run(args))
        elif args.command == 'dpsyn-campaign':
            from .dpsyn_campaign import freeze, execute
            print(execute(freeze(read(args.manifest), args.dpsyn_repo, args.output)))
        elif args.command == 'dpsyn-campaign-report':
            from .dpsyn_campaign import report
            print(report(args.run.resolve()))
        elif args.command == 'dpsyn-profile':
            from .dpsyn import run
            print(run(args))
        elif args.command in {'freeze', 'finalize'}:
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
