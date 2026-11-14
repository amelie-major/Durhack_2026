# Contributing

How we work together during DurHack. We have 24 hours, so the rules are here to stop us stepping on each other and breaking the demo, not to slow us down. If a rule is getting in the way, say so in the group chat and we'll drop it.

## 1. Claim work on the board
- Every chunk of work gets a GitHub Issue. Keep it to one line if that's all it needs.
- Assign yourself **before** you start, so two people don't build the same thing at 3am.
- Labels: `data`, `model`, `backend`, `frontend`, `infra`, `demo`, `bug`.
- Tiny fixes (typos, a config tweak) don't need an Issue. Just mention them in the chat.

## 2. One branch per Issue
- Don't push straight to `main`. `main` is always the version we could demo right now.
- Branch off an up-to-date `main`:

  ```bash
  git checkout main
  git pull
  git checkout -b <type>/<issue-number>-<short-description>
  ```

- Branch name examples:
  - `feat/3-transaction-tokenizer`
  - `feat/6-training-loop`
  - `fix/8-nan-loss`
  - `data/2-synthetic-transactions`
  - `demo/11-streamlit-ui`
- Keep branches short-lived: aim to merge within a few hours. Long-running branches turn into merge hell.

## 3. Commits
- Commit little and often. A commit is also your undo button.
- Format: `<type>: <what changed>`. Types: `feat`, `fix`, `data`, `model`, `demo`, `chore`, `docs`.
  - `feat: add merchant-category embedding`
  - `fix: clip gradients to stop NaN loss`
  - `data: generate 100k synthetic transactions`
  - `demo: show anomaly scores in UI`
- Add the Issue number if there is one: `fix: handle empty sequences (#8)`.
- **Never commit** model checkpoints, datasets, `.env` files or API keys. Put big files in the shared drive and add the path to `.gitignore`.

## 4. Pull requests
- Push your branch and open a PR into `main`. Put `Closes #<issue-number>` in the description.
- Get **one quick approval** from a teammate. Ping them in the chat; reviews should take minutes, not hours.
- The reviewer checks that:
  - it runs on their machine (or at least imports without errors),
  - it doesn't break anything already on `main`,
  - nothing secret or huge is committed.
- Anything that passes those checks gets approved. We're shipping a demo, not a code-quality contest.
- If CI is set up, it must be green. If it isn't, run the smoke test locally (see README).
- **Squash and merge**, then delete the branch.

## 5. Staying in sync
- Pull `main` into your branch regularly, and always before opening a PR:

  ```bash
  git checkout main && git pull
  git checkout <your-branch> && git merge main
  ```

- Fix conflicts locally, check it still runs, then push.
- Stuck on a conflict for more than 10 minutes? Ask for help, don't fight it alone.

## 6. Code and running things
- Run your code locally before pushing (setup and run commands are in the README).
- Add a test where it's cheap and useful (data shapes, tokenizer round-trips, anything that's already broken once). We don't need full coverage.
- Pin new dependencies in `requirements.txt` and tell the group when you add one.
- Hard-coded paths and magic numbers go in `config.py`, not scattered around the code.

## 7. Demo freeze
- **2 hours before submission, `main` is frozen**: bug fixes for the demo only, no new features.
- Tag the build we present: `git tag demo && git push --tags`.
- Whoever presents does a full run-through from a fresh clone at least once before judging.
