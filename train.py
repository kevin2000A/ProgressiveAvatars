"""Run the complete ProgressiveAvatars training recipe with one command."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train from scratch through final refinement")
    parser.add_argument('--source', type=Path, required=True, help='Prepared dataset directory')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the final model')
    parser.add_argument('--subject', default='avatar')
    parser.add_argument('--smoke', action='store_true', help='Check both stages with 500 + 50 steps')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = output / 'base'
    base_steps, final_steps = (500, 550) if args.smoke else (600000, 650000)
    checkpoint = base / f'chkpnt{base_steps}.pth'
    common = [sys.executable, '-m', 'training.run_stage', '--source', str(source),
              '--subject', args.subject]
    if args.smoke:
        common.append('--smoke')
    stages = [
        ('fresh', common + ['--output', str(base)]),
        ('refinement', common + ['--output', str(output), '--checkpoint', str(checkpoint),
                                '--iterations', str(final_steps)]),
    ]
    for name, command in stages:
        print(f'Starting {name} stage', flush=True)
        subprocess.run(command, cwd=repo, check=True)
    receipt = json.loads((output / 'training_complete.json').read_text())
    receipt.update(training_from_scratch=True, total_iterations=final_steps,
                   base_iterations=base_steps, refinement_iterations=final_steps-base_steps)
    (output / 'training_complete.json').write_text(json.dumps(receipt, indent=2))
    print(f'TRAINING_COMPLETE: {output / f"chkpnt{final_steps}.pth"}', flush=True)


if __name__ == '__main__':
    main()
