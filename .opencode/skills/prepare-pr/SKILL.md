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

3. **Commit changes** — if there are unstaged changes, stage them with `git add -u` and commit with `git commit -m "wip"`. This ensures the rebase step can operate cleanly.

4. **Rebase** — run `git rebase main`. If there are conflicts, stop and report them. Do not proceed until conflicts are resolved.

5. **Fix lint** — run `task lint-fix`. If any files were modified, stage them with `git add -u` and amend the WIP commit: `git commit --amend --no-edit`.

6. **Run lint check** — run `task lint` to confirm no remaining issues. If it fails, stop and report them.

7. **Run tests** — run `task test`. If tests fail, stop and report the failures. Do not open a PR with failing tests.

8. **Open a PR** — replace the WIP commit message with a descriptive one: `git commit --amend -m "<descriptive message>"`, then run `gh pr create --fill`. Return the PR URL.
