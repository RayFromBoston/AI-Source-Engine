# Protect `main` (Do This Once)

This file is the exact setup to stop people from breaking `main`.

## What I already set in this repo

- Added `.github/CODEOWNERS` with:
  - `* @RayFromBoston`

Once branch protection is enabled with "Require review from Code Owners",
your review is mandatory for merges to `main`.

## One-time GitHub settings (5 minutes)

Go to:

- **Repo -> Settings -> Rules -> Rulesets -> New branch ruleset**

Use these settings:

### Target

- Branch name pattern: `main`
- Enforcement status: **Active**

### Turn ON these protections

1. **Require a pull request before merging**
   - Required approvals: **1** (or 2 if you want stricter)
   - Dismiss stale approvals when new commits are pushed
   - Require review from Code Owners
   - Require conversation resolution before merge

2. **Require status checks to pass**
   - Select CI checks from `.github/workflows/ci.yml`
   - Require branches to be up to date before merging

3. **Block dangerous pushes**
   - Block force pushes
   - Block branch deletion

### Optional (recommended)

- Require signed commits
- Require linear history
- Restrict who can push to matching branches

## Merge settings (recommended)

Go to:

- **Repo -> Settings -> General -> Pull Requests**

Recommended:

- Allow only **Squash merge**
- Enable **Automatically delete head branches**

## Sanity test after setup

1. Open a small PR to `main`.
2. Confirm GitHub refuses merge until:
   - CI is green
   - CODEOWNER review is approved
3. Confirm direct push to `main` is blocked.

If all 3 are true, `main` is protected correctly.
