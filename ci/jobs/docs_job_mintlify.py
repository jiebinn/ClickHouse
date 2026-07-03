import argparse

from ci.jobs.scripts.docs.check_readonly_copies import check_readonly_copies
from ci.jobs.scripts.docs.mintlify_docs_check import DEFAULT_CHECKS
from ci.praktika.info import Info
from ci.praktika.result import Result
from ci.praktika.utils import Utils


def parse_args():
    parser = argparse.ArgumentParser(description="Mintlify Docs Check Job")
    parser.add_argument("--test", help="Sub check name", default="")
    return parser.parse_args()


def _readonly_copies_guard():
    # One-way sync: fail edits to docs folders whose canonical source lives in
    # another repo (declared in ci/jobs/scripts/docs/readonly_copies.json). This
    # is aggregator-only -- the consuming repos are the source of truth, so this
    # is deliberately not part of the shared DEFAULT_CHECKS.
    changed_files = Info().get_changed_files()
    if changed_files is None:
        # Fail close: without the changed-file list we cannot prove the PR does
        # not touch read-only copies, so do not report success.
        print("Error: the changed-file list is unavailable, cannot run the check.")
        return False
    return check_readonly_copies(changed_files)


if __name__ == "__main__":

    args = parse_args()
    testpattern = args.test

    docs_dir = f"{Utils.cwd()}/docs"

    def selected(name):
        # Case-insensitive substring match; an empty pattern runs every check.
        return testpattern.lower() in name.lower()

    # The mint check definitions are shared with the standalone driver
    # (ci/jobs/scripts/docs/mintlify_docs_check.py). This job already runs inside
    # the docs-builder image with the docs present natively, so it runs them
    # directly; add new checks to DEFAULT_CHECKS, not here. --test selects a
    # subset by sub-check name.
    results = [
        Result.from_commands_run(name=name, command=command, workdir=docs_dir)
        for name, command in DEFAULT_CHECKS
        if selected(name)
    ]

    # The read-only guard runs from the repo root (not the docs root), so keep
    # it out of the docs_dir loop above.
    if selected("No direct edits to read-only copied docs"):
        results.append(
            Result.from_commands_run(
                name="No direct edits to read-only copied docs",
                command=_readonly_copies_guard,
            )
        )

    Result.create_from(results=results).complete_job()