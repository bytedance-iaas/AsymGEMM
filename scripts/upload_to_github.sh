#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/upload_to_github.sh [options]

Upload the current AsymGEMM working tree to a standalone GitHub repository by
creating a temporary git repo from the local files and pushing that snapshot.

Options:
  --repo URL          Target remote URL.
                      Default: git@github.com:bytedance-iaas/AsymGEMM.git
  --branch NAME       Target branch name. Default: main
  --message TEXT      Commit message for the uploaded snapshot.
                      Default: Initial import from local AsymGEMM checkout
  --force             Force-push the target branch.
  --dry-run           Prepare the temporary repo but do not push.
  -h, --help          Show this help message.

The script also reads ignore patterns from .gitignore and .gitignore.upload in
the repository root when building the upload snapshot.
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
            --exclude='.github/workflows/*.tmp' \
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

repo_url="git@github.com:bytedance-iaas/AsymGEMM.git"
branch_name="main"
commit_message="Initial import from local AsymGEMM checkout"
force_push=0
dry_run=0

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
            force_push=1
            shift
            ;;
        --dry-run)
            dry_run=1
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
require_cmd mktemp
require_cmd tar

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/asymgemm-upload.XXXXXX")
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

snapshot_dir="$tmp_dir/snapshot"
mkdir -p "$snapshot_dir"

echo "Creating snapshot from $repo_root"
copy_tree "$repo_root" "$snapshot_dir"

cd "$snapshot_dir"
git init --initial-branch="$branch_name" >/dev/null
git add -A

if git diff --cached --quiet; then
    echo "Error: snapshot is empty, nothing to upload." >&2
    exit 1
fi

if ! git config user.name >/dev/null; then
    git config user.name "${GIT_AUTHOR_NAME:-AsymGEMM Uploader}"
fi
if ! git config user.email >/dev/null; then
    git config user.email "${GIT_AUTHOR_EMAIL:-asymgemm-uploader@local}"
fi

git commit -m "$commit_message" >/dev/null
git remote add origin "$repo_url"

echo "Prepared temporary repository in $snapshot_dir"
echo "Target remote: $repo_url"
echo "Target branch: $branch_name"

if ((dry_run)); then
    echo "Dry run enabled. Inspect the snapshot above; no push was performed."
    trap - EXIT
    exit 0
fi

if ((force_push)); then
    echo "Force-pushing snapshot to origin/$branch_name"
    git push --force origin "HEAD:$branch_name"
else
    echo "Pushing snapshot to origin/$branch_name"
    git push --set-upstream origin "$branch_name"
fi
