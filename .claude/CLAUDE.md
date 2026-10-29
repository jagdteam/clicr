<!-- fleet:autonomy:start -->
## Autonomy rules (fleet)

- Work only on a branch whose name starts with `claude/`. Never work on or push to `main` or `master`.
- Never read or print secrets or `.env` files.
- Before every push, run this repo's verify command (tests and/or build). Push only if it passes.
- Say "done" only when the task's written stop condition is checkably met. Show the check.
- If a fact is missing, write the question into `BLOCKED.md` at the repo root and move on to other work. Do not guess.
- This repo may lack GitHub branch protection; a push-guard hook blocks pushes to main/master, force-pushes and deletes. Never try to work around it.
- Stay inside this repo. Do not touch anything outside it.
- When the task's stop condition is met, open a draft PR whose body quotes the stop condition and the check that proves it. Pushing a branch without a PR is not done.
- Before marking a PR ready for review, run the reviewer pass named in the brief and fix or list every finding in the PR body.
<!-- fleet:autonomy:end -->
