#!/usr/bin/env python3
"""
Unit tests for the pure helpers in `create_release.py`. Run as part of the
`ci/tests/` suite with `pytest ci/tests/test_create_release.py` from the repo
root.
"""

import os
import sys
import unittest

# `create_release` lives under `tests/ci` and imports sibling modules by bare
# name, so put that directory on `sys.path` only while importing it and remove
# it again afterwards to avoid leaking it into the rest of the pytest session.
_CI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tests", "ci")
)
sys.path.insert(0, _CI_DIR)
try:
    # pylint: disable=import-error
    from create_release import ReleaseInfo
finally:
    sys.path.remove(_CI_DIR)


class TestIsEmptyPatchRelease(unittest.TestCase):
    def test_rejects_empty_rerun(self):
        # Already-published branch: the only commit since the previous
        # stable/lts tag is the automated version bump (e.g. v25.8.28.1-lts).
        self.assertTrue(ReleaseInfo._is_empty_patch_release(patch=28, tweak=1))

    def test_allows_first_release_of_new_branch(self):
        # First user-facing stable/lts release on a freshly cut branch. Its
        # previous tag is vX.Y.1.1-new and the single testing -> stable commit
        # also yields tweak == 1, but the release is legitimate.
        self.assertFalse(ReleaseInfo._is_empty_patch_release(patch=1, tweak=1))

    def test_allows_non_empty_patch_release(self):
        # Real commits on top of the previous release -> tweak > 1.
        self.assertFalse(ReleaseInfo._is_empty_patch_release(patch=28, tweak=42))

    def test_allows_non_empty_first_release(self):
        self.assertFalse(ReleaseInfo._is_empty_patch_release(patch=1, tweak=2222))


if __name__ == "__main__":
    unittest.main()
