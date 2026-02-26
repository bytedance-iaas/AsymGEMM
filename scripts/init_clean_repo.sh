#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/init_clean_repo.sh [options]

Create a brand-new local git repository from the current AsymGEMM working tree,
excluding files matched by .gitignore and .gitignore.upload.

Options:
  --dest DIR          Destination directory for the clean repo.
                      Default: ../AsymGEMM_clean
  --repo URL          Optional git remote to add as origin.
                      Default: git@github.com:bytedance-iaas/AsymGEMM.git
  --branch NAME       Initial branch name. Default: main
  --message TEXT      Initial commit message.
                      Default: Initial import of AsymGEMM
  --no-remote         Do not add origin.
  -h, --help          Show this help message.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

copy_tree() {
    local source_dir=$1
    local dest_dir=$2
    local upload_ignore="$source_dir/.gitignore.upload"

    if command -v rsync >/dev/null 2>&1; then
        rsync -a \
            --filter=':- .gitignore' \
            --filter=':- .gitignore.upload' \
            --exclude='.git/' \
            --exclude='.gitmodules' \
            "$source_dir"/ "$dest_dir"/
        return
    fi

    local tar_args=(
        --exclude='.git'
        --exclude='.gitmodules'
    )

    if [[ -f "$upload_ignore" ]]; then
        while IFS= read -r pattern; do
            [[ -z "$pattern" || "$pattern" =~ ^[[:space:]]*# ]] && continue
            tar_args+=("--exclude=$pattern")
        done < "$upload_ignore"
    fi

    tar "${tar_args[@]}" -C "$source_dir" -cf - . | tar -C "$dest_dir" -xf -
}

dest_dir="../AsymGEMM_clean"
repo_url="git@github.com:bytedance-iaas/AsymGEMM.git"
branch_name="main"
commit_message="Initial import of AsymGEMM"
add_remote=1

while (($# > 0)); do
    case "$1" in
        --dest)
            dest_dir=$2
            shift 2
            ;;
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
        --no-remote)
            add_remote=0
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

require_cmd git
require_cmd mkdir
require_cmd rm
require_cmd tar

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
dest_dir=$(realpath -m "$repo_root/$dest_dir")

if [[ -e "$dest_dir" ]]; then
    echo "Error: destination already exists: $dest_dir" >&2
    exit 1
fi

mkdir -p "$dest_dir"

echo "Creating clean repository at $dest_dir"
copy_tree "$repo_root" "$dest_dir"

cd "$dest_dir"
rm -rf .git
git init --initial-branch="$branch_name" >/dev/null
git add -A

if git diff --cached --quiet; then
    echo "Error: clean repository is empty, nothing to commit." >&2
    exit 1
fi

if ! git config user.name >/dev/null; then
    git config user.name "${GIT_AUTHOR_NAME:-AsymGEMM Uploader}"
fi
if ! git config user.email >/dev/null; then
    git config user.email "${GIT_AUTHOR_EMAIL:-asymgemm-uploader@local}"
fi

git commit -m "$commit_message" >/dev/null

if ((add_remote)); then
    git remote add origin "$repo_url"
fi

echo "Initialized clean git repository."
echo "Location: $dest_dir"
echo "Branch: $branch_name"

if ((add_remote)); then
    echo "Origin: $repo_url"
    echo "Next step: cd $dest_dir && git push -u origin $branch_name"
else
    echo "Next step: cd $dest_dir"
fi
