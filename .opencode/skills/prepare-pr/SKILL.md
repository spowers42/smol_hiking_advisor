---
name: prepare-pr
description: Create a branch, fix lint, run tests, and open a PR.
---

## Steps

1. **Check current branch** — run `git branch --show-current`.
   - If on `main`, always create a new branch — ask the user for a branch name.
   - If on a feature branch, ask the user: "We're on `{branch}`. Continue on this branch, or create a new one?"
     - **Continue** — skip to step 2.
     - **New branch** — ask for a name, then create and switch to it.

2. **Sync `main`** — run `git fetch origin main && git checkout main && git pull --ff-only origin main && git checkout -`. If the pull fails, stop and report that `main` has diverged.

3. **Rebase** — run `git rebase main`. If there are conflicts, stop and report them. Do not proceed until conflicts are resolved.

4. **Fix lint** — run `task lint-fix`.

5. **Run tests** — run `task test`. If tests fail, stop and report the failures. Do not open a PR with failing tests.

6. **Commit and open a PR** — if there are staged changes, commit them with a descriptive message, then run `gh pr create --fill`. If nothing is staged, commit the lint fixes with a descriptive message, then open the PR. Return the PR URL.
