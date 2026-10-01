"""Prepare strongest remaining saved cross-split candidates against verified v4."""
from pathlib import Path
from .apply_ruod_visual_families import read, write
from .review_ruod_v3_candidates import compose_actions
from .review_ruod_next_families import prepare

if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    output = repo / 'outputs/ruod-near-review-batch04-v4'
    batches = [read(repo / ('outputs/' + folder + '/proposed_actions.json')) for folder in
               ('ruod-visual-family-proposal', 'ruod-visual-family-proposal-batch02',
                'ruod-visual-family-proposal-batch03')]
    write(output / 'v4_cumulative_actions.json', compose_actions(batches))
    prepare(repo / 'outputs/ruod-near-duplicate-screen', output / 'v4_cumulative_actions.json',
            output, prefix='B04', max_groups=12,
            boundaries=('train__test', 'validation__test', 'train__validation'))
