import io
import unittest
import numpy as np
from PIL import Image
from tools.screen_ruod_near_duplicates import hashes, screen_pairs


class NearDuplicateTests(unittest.TestCase):
    def test_fingerprint_determinism_and_recompression(self):
        y, x = np.mgrid[:128, :128]
        image = Image.fromarray(np.stack([(x*2)%256, (y*2)%256, ((x+y)*3)%256], axis=-1).astype('uint8'))
        first = hashes(image)
        self.assertEqual(first, hashes(image.copy()))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        with Image.open(io.BytesIO(buffer.getvalue())) as compressed:
            second = hashes(compressed)
        distances = [bin(int(a,16)^int(b,16)).count('1') for a,b in zip(first,second)]
        self.assertLessEqual(distances[0], 8)

    def test_candidates_are_bounded_ranked_and_not_mutated(self):
        left = [dict(id=1, phash='0000000000000000', dhash='0000000000000000')]
        right = [dict(id=i, phash=format(i,'016x'), dhash=format(i,'016x')) for i in range(20)]
        candidates, total = screen_pairs(left, right, keep=3)
        self.assertEqual(total, 20)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]['right']['id'], 0)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 20)
        self.assertEqual([c['score'] for c in candidates], sorted(c['score'] for c in candidates))


if __name__ == '__main__':
    unittest.main()
