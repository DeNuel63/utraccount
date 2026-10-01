"""Present 12 strongest test-linked priority groups; no split changes."""
from pathlib import Path
from .apply_ruod_visual_families import read, write, digest_file
from .review_ruod_v3_candidates import compose_actions
from .review_ruod_next_families import prepare

if __name__ == '__main__':
    repo=Path(__file__).resolve().parents[1]
    output=repo/'outputs/ruod-near-review-batch05-v5'
    folders=['ruod-visual-family-proposal']+['ruod-visual-family-proposal-batch'+s for s in ('02','03','04')]
    actions=compose_actions([read(repo/'outputs'/f/'proposed_actions.json') for f in folders])
    write(output/'v5_cumulative_actions.json',actions)
    screen=repo/'outputs/ruod-v5-workload'
    prepare(screen,output/'v5_cumulative_actions.json',output,prefix='B05',max_groups=12,
            boundaries=('train__test','validation__test','train__validation'),
            max_score=4,test_linked_only=True)
    write(output/'selection.json',dict(base_split='v5-reviewed',max_combined_distance=4,
        test_linked_only=True,group_limit=12,workload_summary_sha256=digest_file(screen/'summary.json'),
        limitation='Closure includes only priority edges; weaker links and additional family members remain unreviewed.'))
