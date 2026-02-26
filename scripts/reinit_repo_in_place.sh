#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/reinit_repo_in_place.sh [options]

Reinitialize the current directory as a brand-new git repository with no prior
history, using the current working tree as the source.

This script is destructive:
  - it removes the current .git directory
  - it removes and re-adds third-party submodule directories

Options:
  --repo URL          Remote URL to add as origin.
                      Default: git@github.com:bytedance-iaas/AsymGEMM.git
  --branch NAME       Initial branch name. Default: main
  --message TEXT      Initial commit message.
                      Default: Initial import of AsymGEMM
  --force             Required. Acknowledge the destructive reset.
  -h, --help          Show this help message.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

repo_url="git@github.com:bytedance-iaas/AsymGEMM.git"
branch_name="main"
commit_message="Initial import of AsymGEMM"
force_reset=0

while (($# > 0)); do
    case "$1" in
        --repo)
            repo_url=$2
            shift 2
            ;;
        --branch)
            branch_name=$2
            shift 2
            ;;
        --message)
            commit_message=$2
            shift 2
            ;;
        --force)
            force_reset=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ((force_reset == 0)); then
    echo "Error: --force is required because this deletes the current .git history." >&2
    exit 1
fi

require_cmd git
require_cmd rm
require_cmd mktemp

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

if [[ ! -f .gitmodules ]]; then
    echo "Error: .gitmodules is required to recreate third-party submodules." >&2
    exit 1
fi

submodule_names=()
submodule_paths=()
submodule_urls=()
while IFS=$'\n' read -r line; do
    name=${line%% *}
    path=${line#* }
    url=$(git config -f .gitmodules --get "submodule.${name#submodule.}.url")
    submodule_names+=("$name")
    submodule_paths+=("$path")
    submodule_urls+=("$url")
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')

if ((${#submodule_paths[@]} == 0)); then
    echo "Error: no submodules found in .gitmodules." >&2
    exit 1
fi

echo "Reinitializing git metadata in $repo_root"
if [[ -e .git ]]; then
    rm -rf .git
else
    existing_git_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "$existing_git_root" && "$existing_git_root" != "$repo_root" ]]; then
        echo "No local .git found. This directory is currently inside parent repo: $existing_git_root"
        echo "A new standalone .git repository will be created in $repo_root"
    fi
fi

for path in "${submodule_paths[@]}"; do
    if [[ -e "$path/.git" ]]; then
        rm -rf "$path/.git"
    fi
    rm -rf "$path"
done

git init --initial-branch="$branch_name" >/dev/null

cat > .git/info/exclude <<'EOF'
tests/
demo/
stubs/
scripts/
build/
dist/
*.egg-info/
__pycache__/
*.pyc
.cache/
.idea/
.vscode/
EOF

git add -A

if git diff --cached --quiet; then
    echo "Error: initial repository is empty, nothing to commit." >&2
    exit 1
fi

if ! git config user.name >/dev/null; then
    git config user.name "${GIT_AUTHOR_NAME:-AsymGEMM Uploader}"
fi
if ! git config user.email >/dev/null; then
    git config user.email "${GIT_AUTHOR_EMAIL:-asymgemm-uploader@local}"
fi

git commit -m "$commit_message" >/dev/null

for i in "${!submodule_paths[@]}"; do
    git submodule add "${submodule_urls[$i]}" "${submodule_paths[$i]}"
done

git remote add origin "$repo_url"

echo "Initialized a fresh repository in place."
echo "Branch: $branch_name"
echo "Origin: $repo_url"
echo "Next step: git push -u origin $branch_name"
