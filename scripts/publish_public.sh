#!/bin/bash
# Publish the current main tree to the public GitHub repo as ONE new commit on the squashed `public` history, and the
# rendered Quarto book to gh-pages. Full history stays local. Refuses if private identifiers appear (patterns come from
# every value in the untracked settings.local.yaml).
# Usage: scripts/publish_public.sh "commit message"
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
msg=${1:?commit message}
E=30431939+yubrajbhandari923@users.noreply.github.com
pat=$(python3 -c "import re;t=open('settings.local.yaml').read();v=re.findall(r'^\s+\w+:\s*(\S+)',t,re.M);print('|'.join(re.escape(x) for x in v))")
if git grep -nE "$pat" main -- . ':!uv.lock'; then echo "REFUSING: private identifiers in main (above)"; exit 1; fi
(cd reports/quarto && uv run python build.py >/dev/null)
if grep -rlE "$pat" reports/quarto/_build; then echo "REFUSING: private identifiers in rendered site"; exit 1; fi
id() { GIT_AUTHOR_NAME=yubrajbhandari923 GIT_AUTHOR_EMAIL=$E GIT_COMMITTER_NAME=yubrajbhandari923 GIT_COMMITTER_EMAIL=$E "$@"; }
git fetch -q origin main gh-pages
if [ "$(git rev-parse 'main^{tree}')" != "$(git rev-parse 'origin/main^{tree}')" ]; then
  c=$(id git commit-tree 'main^{tree}' -p origin/main -m "$msg")
  git branch -f public "$c" && git push origin public:main
fi
touch reports/quarto/_build/.nojekyll
idx=$(mktemp -u); GIT_INDEX_FILE=$idx git --work-tree=reports/quarto/_build add -A . ; t=$(GIT_INDEX_FILE=$idx git write-tree); rm -f "$idx"
if [ "$t" != "$(git rev-parse 'origin/gh-pages^{tree}')" ]; then
  p=$(id git commit-tree "$t" -p origin/gh-pages -m "Rendered report: $msg")
  git branch -f gh-pages "$p" && git push origin gh-pages
fi
echo "published: https://github.com/yubrajbhandari923/pravah  https://yubrajbhandari923.github.io/pravah/"
