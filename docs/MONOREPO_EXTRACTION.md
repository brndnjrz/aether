# Extracting a Project From a Monorepo Into Its Own Repo

A reusable recipe for pulling one subdirectory out of a larger git repo (like this
workspace's `Finance` monorepo) and turning it into its own standalone GitHub repo —
with a clean, fresh history rather than dragging the monorepo's commit log along.
This is exactly what was done to turn `Finance/stock-app/stock-edge-pro` into the
standalone `aether` repo.

## When to use this

- A project living inside a personal monorepo has grown into something you want to
  push to GitHub, deploy, or share on its own.
- You don't need (or don't want) the monorepo's commit history — you're fine
  starting the new repo at commit 1.
- You want the monorepo to fully stop tracking the subdirectory going forward,
  without deleting anything from disk.

If you *do* want to preserve history for just that subdirectory, use
`git subtree split` instead of this recipe — see [Alternative](#alternative-preserving-history)
below.

## Prerequisites checklist

Before touching git, make sure the subdirectory won't leak anything you don't want
public (or just don't want cluttering a fresh repo):

- [ ] `.env` / secrets are not hardcoded anywhere — only referenced via env vars
- [ ] Any local databases, trained models, logs, or other runtime artifacts are
      identified (they'll need their own ignore rule)
- [ ] Any editor/tool-specific settings (`.claude/`, `.vscode/`, etc.) are identified
- [ ] You know whether the project has real personal/financial/sensitive data sitting
      in a tracked file (not just runtime data — check markdown docs, fixtures, etc.
      too)

## Step 1 — Write a `.gitignore` inside the project directory

The monorepo's root `.gitignore` won't exist once this becomes its own repo, so the
project needs its own before its first commit:

```bash
cd path/to/monorepo/some-dir/my-project
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
venv/
.venv/
.env
.env.*
!.env.example
.DS_Store
.claude/settings.local.json
storage/*
!storage/.gitkeep
EOF
```

Adjust the runtime-data lines (`storage/*` here) to whatever the project actually
generates — trained models, SQLite databases, log files, etc.

## Step 2 (optional) — Rename the directory

If you're adopting a new project name at the same time, rename the folder now, before
extracting it, so the new repo starts life with the right name:

```bash
mv path/to/monorepo/some-dir/old-name path/to/monorepo/some-dir/new-name
```

A plain `mv` is fine even though the directory is git-tracked — Step 3 removes it from
the monorepo's index entirely, so there's no rename to preserve there. If you rename,
grep the whole workspace for the old name afterward and fix any docs, `.gitignore`
rules, or CI config that reference the old path:

```bash
grep -rln "old-name" . --exclude-dir=.git
```

## Step 3 — Stop the monorepo from tracking it

In the monorepo's root:

```bash
git rm -r --cached path/to/some-dir/old-name   # or new-name if you didn't rename
```

`--cached` removes the files from git's index only — it does not touch the working
tree, so nothing on disk is deleted. (If you renamed in Step 2, git already sees the
old path's files as deleted from disk; `git rm --cached` on that path just tells git
to stop expecting them back.)

Then tell the monorepo to ignore the directory going forward, so it can never get
re-added by accident (e.g. via a stray `git add -A`):

```bash
echo "some-dir/new-name/" >> .gitignore
git add .gitignore
```

Commit the removal:

```bash
git commit -m "Move some-dir/new-name out of this repo into its own standalone repo"
```

This is the one step that touches the monorepo's shared history — everything before
and after this operates on the extracted project in isolation.

## Step 4 — Turn the subdirectory into its own repo

```bash
cd path/to/monorepo/some-dir/new-name
git init
git add -A
```

**Before committing, verify nothing unwanted got staged** — this is the actual safety
check, not just the `.gitignore` file existing:

```bash
git status --short | grep -E "\.env$|storage/|__pycache__|\.claude"
```

This should print nothing (or only intentional placeholder files like
`storage/.gitkeep`). If it prints anything else, fix `.gitignore` and re-run
`git add -A` before continuing.

```bash
git commit -m "Initial commit — <project name>"
git branch -M main
```

## Step 5 — Create the empty GitHub repo

Via the web UI (or `gh repo create` if you have the CLI installed):

1. Go to **github.com/new**
2. Set the repository name
3. Leave it **completely empty** — do not add a README, `.gitignore`, or license from
   GitHub's side. If GitHub creates any files, its `main` branch will have a commit
   that conflicts with the one you already made locally in Step 4.
4. Choose Public or Private
5. Create it

## Step 6 — Connect and push

```bash
git remote add origin git@github.com:<username>/<repo-name>.git
git push -u origin main
```

Done. The project is now a standalone repo tracking `origin/main`, and the monorepo no
longer has any record of it beyond the one removal commit from Step 3.

## Worked example (this repo)

The exact commands run to produce this repo from `Finance/stock-app/stock-edge-pro`:

```bash
# Inside stock-app/stock-edge-pro (before renaming)
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
*.pyo
venv/
.venv/
env/
.env
.env.*
!.env.example
.DS_Store
.vscode/
.idea/
.claude/settings.local.json
storage/*
!storage/.gitkeep
EOF

# Rename, then fix references to the old name across the workspace
mv stock-app/stock-edge-pro stock-app/aether
grep -rln "stock-edge-pro" . --exclude-dir=.git --exclude-dir=_reference

# Back in the Finance monorepo root
git rm -r --cached stock-app/stock-edge-pro --quiet
# .gitignore edited to add: stock-app/aether/
git add .gitignore
git commit -m "Move stock-app/aether out of this repo into its own standalone repo"

# Inside stock-app/aether
git init
git add -A
git status --short | grep -E "\.env$|storage/|__pycache__|\.claude"   # printed nothing unexpected
git commit -m "Initial commit — Aether"
git branch -M main

# After creating an empty github.com/brndnjrz/aether repo via the web UI
git remote add origin git@github.com:brndnjrz/aether.git
git push -u origin main
```

## Alternative: preserving history

If a future project's commit history is actually worth keeping, use
`git subtree split` instead of `git rm --cached` + fresh `git init`:

```bash
# From the monorepo root — rewrites history to include only commits touching that path
git subtree split -P some-dir/project-name -b split-branch

# Push just that branch to the new (empty) remote
git push git@github.com:<username>/<repo-name>.git split-branch:main
```

This preserves authorship and commit messages for that subdirectory's history, but
takes longer on a large monorepo and can pull in unrelated context (commits that
touched the path incidentally). The fresh-`git init` approach above is simpler and was
the right call for this project since a clean start was preferred over dragging along
the monorepo's commit log.
