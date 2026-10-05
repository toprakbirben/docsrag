# Design: `stuff changes` / `stuff why` — explain what I added, from diffs + commit/PR messages

## Context
`stuff sync projects` (2026-10-05) lets stuffrag answer questions about the *current* code in `~/projects`. It knows history only as the last 15 commit subjects in each project's `__overview__`. Toprak wants to ask:
1. **What changed in a period:** "what did I add to fuutball last week?"
2. **When and why was X added:** "when did I add rate limiting to fuutball, and why?"
3. Answers that **combine the commit/PR message with the actual diff** to say what was added, removed or changed, and where the two disagree.

Uncommitted work is out of scope. PR descriptions come from GitHub via `gh`: read-only, stored nowhere but the answer.

## Decision: on-demand commands, nothing indexed
Git and code do every lookup (commit selection, date ranges, finding the introducing commit). The LLM only explains (Rule 5). Historical diffs are never written to the database. History can still contain secrets that current files no longer have (committed, then removed), so keeping it out of the index keeps the exposure small. Rejected: indexing every commit as a document. Retrieval can't return "everything in a date range" completely, and it would store old diffs.

## Commands
- `stuff changes <project> [--since "1 week ago"] [--until <date>]`
- `stuff why <project> "<topic>"`

`<project>` is a folder name under `~/projects`. A project skipped by `SKIP_PROJECTS`, or one without `.git`, is a clear error.

## `changes`
1. **Select:** `git log --first-parent --since --until`. Each merged PR appears once (its merge commit) and direct commits appear individually. `--since`/`--until` are passed to git verbatim, so git parses relative dates; no LLM date parsing.
2. **Diff:** `git diff <sha>^1 <sha>`, so a merge yields the full PR diff (`git show <sha>` for a root commit). The diff is split per file.
3. **PR text:** one `gh pr list -R <origin> --state merged --limit 500 --json number,title,body,mergeCommit` per run, mapped by `mergeCommit.oid` (works for merge and squash merges). If there's no GitHub origin, `gh` is missing or unauthenticated, or it errors, print one warning and continue with commit messages only.
4. **Secret safety** (reuses `ingest/projects.py`):
   - A per-file diff is dropped if `_allowed(path)` is false (lockfiles, `docs/`, `vendor/`, binaries, non-allow-listed extensions) or `is_secret(path, file_diff)` is true.
   - Commit messages and PR bodies matching `is_secret` are replaced by `(omitted: matched a secret pattern)`.
   - Output always states `N files omitted as secret, M hidden/non-code`, as paths only.
5. **Budget:** each commit's diff is capped at 6,000 chars. If the total context exceeds 40,000 chars, the largest commits are reduced to `--stat` lines until it fits. The output lists which commits were reduced (fail loud).
6. **LLM:** a system prompt instructs it to use only the given messages, PR descriptions and diffs; write **Added / Removed / Changed**; cite commits as `[abc1234]`; and explicitly flag any mismatch between a message and its diff. It reuses `generate.chat` with the config's LLM.
7. **Output:** the answer, then `Commits:` lines (`sha date subject [PR #n]`) and the omission/reduction notes.

## `why`
1. `retrieve(conn, topic, cfg, project)` gets the top chunks (the existing `--project` filter).
2. For each of the top 3 chunks: strip the `project/path` label line; pick the longest line with at least 20 non-space chars, excluding comment-only lines; run `git log -S "<line>" --reverse --format=%H -- <path>`; the first SHA is the introducing commit. Overview docs and chunks with no qualifying line are skipped.
3. Deduplicate commits (max 3). For each, gather message, PR text (same `gh` map) and the diff for that file, with the same secret/allow-list filtering and caps.
4. LLM: say when it was added, why (from message/PR), and what the change did, citing commits and the file it found. If no introducing commit is found, say so and do not guess.

## Error handling
Unknown project, no `.git`, or an empty range each produce a plain message and exit code 1. `gh` failure only warns. git failures raise with git's stderr.

## Testing (temp git repo fixture; `gh` and the LLM mocked)
- The date range is honoured, and `--first-parent` makes a merged branch one unit whose diff covers the whole branch.
- PR text attaches to the matching merge SHA, and a `gh` failure warns but still answers.
- A file whose diff adds and later removes an `AKIA…` key is omitted and counted. A secret in a commit message is replaced.
- Budget overflow reduces the largest commit to stat and reports it.
- `why` finds the commit that introduced a line, not a later one that touched the file.
- The prompt sent to the LLM contains messages and diffs and never contains the omitted secret.

Manual check: `stuff changes fuutball --since 2026-09-20 --until 2026-09-23` (PR #123 tactics/stamina) and `stuff changes mtt --since 2026-07-01 --until 2026-07-31` (CRDT work), judged against the actual commits.

## Files
- new `stuffrag/history.py`, `tests/test_history.py`
- edit `stuffrag/cli.py` (2 commands)
