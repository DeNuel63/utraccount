"""Present the strongest unresolved saved test-overlap groups against v3."""
import argparse
from pathlib import Path
from .apply_ruod_visual_families import read, write
from .review_ruod_next_families import prepare


def compose_actions(action_batches):
    final={}
    for batch in action_batches:
        for action in batch:
            split, image_id=action['member'].split(':')
            # Train and validation IDs share the original training namespace.
            key=('test' if split=='test' else 'train')+':'+image_id
            final[key]=dict(action,member=key)
    return list(final.values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--first-actions',type=Path,required=True)
    parser.add_argument('--second-actions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    actions=compose_actions([read(args.first_actions),read(args.second_actions)])
    write(args.output/'v3_cumulative_actions.json',actions)
    prepare(args.screen,args.output/'v3_cumulative_actions.json',args.output,prefix='B03',max_groups=12)
