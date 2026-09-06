#!/usr/bin/env bash
# Export every issue and pull request of the GitHub repo -- bodies, comments,
# reviews and PR commit messages included -- to a directory of JSON files plus
# two flat markdown files (issues.md, prs.md) that read top to bottom. This is
# what the "How this got built" section of the README was written from.
#
# Needs: gh authenticated to github.com (`gh auth status`), jq.
#
#   scripts/export_issues.sh
#   REPO=github.com/murukulu/airglow OUT=~/scratch/airglow-issues scripts/export_issues.sh
#   FORCE=1 scripts/export_issues.sh    # re-fetch files that already exist
#
# Per-item fetches are skipped when their file exists, so an interrupted run
# can simply be re-run. The two list files and the markdown are always rewritten.
set -euo pipefail

REPO=${REPO:-github.com/murukulu/airglow}
OUT=${OUT:-$HOME/scratch/airglow-issues}
FORCE=${FORCE:-0}

# `gh -R` takes HOST/OWNER/REPO; `gh api` wants the host separately.
case $REPO in
    */*/*) HOST=${REPO%%/*}; REPO_PATH=${REPO#*/} ;;
    *) HOST=github.com; REPO_PATH=$REPO ;;
esac

mkdir -p "$OUT"

fetch() { # fetch <dest> <gh args...> -- atomic: no partial file left on interrupt
    local dest=$1
    shift
    if [ -e "$dest" ] && [ "$FORCE" != 1 ]; then
        echo "skip (exists): $dest"
        return
    fi
    echo "fetching $dest"
    gh "$@" > "$dest.part"
    mv "$dest.part" "$dest"
}

# --- issues ------------------------------------------------------------------
# `gh issue list` excludes PRs; those come below.
gh issue list -R "$REPO" --state all -L 500 \
    --json number,title,state,createdAt,closedAt,updatedAt,labels,milestone,author,url \
    > "$OUT/issues.json"

for n in $(jq -r '.[].number' "$OUT/issues.json"); do
    fetch "$OUT/issue-$n.json" issue view "$n" -R "$REPO" \
        --json number,title,state,createdAt,closedAt,body,labels,milestone,author,url,comments
done

# --- pull requests -----------------------------------------------------------
gh pr list -R "$REPO" --state all -L 500 \
    --json number,title,state,createdAt,mergedAt,closedAt,headRefName,author,url \
    > "$OUT/prs.json"

for n in $(jq -r '.[].number' "$OUT/prs.json"); do
    fetch "$OUT/pr-$n.json" pr view "$n" -R "$REPO" \
        --json number,title,state,createdAt,mergedAt,closedAt,body,headRefName,author,url,comments,reviews,commits,closingIssuesReferences
done

# --- milestones --------------------------------------------------------------
if ! gh api --hostname "$HOST" "repos/$REPO_PATH/milestones?state=all&per_page=100" > "$OUT/milestones.json"; then
    echo "warning: milestones unavailable, writing []" >&2
    echo '[]' > "$OUT/milestones.json"
fi

# --- flat markdown -----------------------------------------------------------
{
    echo "# $REPO -- issues"
    for n in $(jq -r 'sort_by(.number) | .[].number' "$OUT/issues.json"); do
        jq -r '
            "\n## #\(.number) \(.title)\n\n" +
            "_\(.state | ascii_downcase), opened \(.createdAt[:10])" +
            (if .closedAt then ", closed \(.closedAt[:10])" else "" end) +
            (if (.labels | length) > 0 then ", labels: \([.labels[].name] | join(", "))" else "" end) +
            (if .milestone then ", milestone: \(.milestone.title)" else "" end) +
            "_  <\(.url)>\n\n" +
            (.body // "" | if . == "" then "_(no body)_" else . end) + "\n" +
            ([.comments[]? |
                "\n### comment by @\(.author.login) (\(.createdAt[:10]))\n\n\(.body)\n"
            ] | join(""))
        ' "$OUT/issue-$n.json"
    done
} > "$OUT/issues.md"

{
    echo "# $REPO -- pull requests"
    for n in $(jq -r 'sort_by(.number) | .[].number' "$OUT/prs.json"); do
        jq -r '
            "\n## PR #\(.number) \(.title)\n\n" +
            "_\(.state | ascii_downcase), branch \(.headRefName), opened \(.createdAt[:10])" +
            (if .mergedAt then ", merged \(.mergedAt[:10])" else "" end) +
            (if (.closingIssuesReferences | length) > 0
                then ", closes \([.closingIssuesReferences[] | "#\(.number)"] | join(" "))"
                else "" end) +
            "_  <\(.url)>\n\n" +
            (.body // "" | if . == "" then "_(no body)_" else . end) + "\n" +
            ([.commits[]? |
                "\n- `\(.oid[:7])` \(.messageHeadline)" +
                (if (.messageBody // "") != "" then "\n\n  " + (.messageBody | gsub("\n"; "\n  ")) else "" end)
            ] | join("")) + "\n" +
            ([.reviews[]? | select((.body // "") != "") |
                "\n### review by @\(.author.login) (\(.state | ascii_downcase), \(.submittedAt[:10]))\n\n\(.body)\n"
            ] | join("")) +
            ([.comments[]? |
                "\n### comment by @\(.author.login) (\(.createdAt[:10]))\n\n\(.body)\n"
            ] | join(""))
        ' "$OUT/pr-$n.json"
    done
} > "$OUT/prs.md"

echo
echo "$(jq length "$OUT/issues.json") issues, $(jq length "$OUT/prs.json") PRs, $(jq length "$OUT/milestones.json") milestones -> $OUT"
echo "read: $OUT/issues.md  $OUT/prs.md"
