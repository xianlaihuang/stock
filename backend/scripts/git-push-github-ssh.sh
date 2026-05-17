#!/usr/bin/env bash
# 通过 SSH 将本地改动提交并推送到 GitHub
# 用法:
#   ./git-push-github-ssh.sh "提交说明"
#   ./git-push-github-ssh.sh -m "提交说明" -b main   # 覆盖默认主分支
#   ./git-push-github-ssh.sh --dry-run

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-master}"
COMMIT_MSG=""
TARGET_BRANCH=""
DRY_RUN=0
ADD_ALL=1
PATHS=()

usage() {
  cat <<'EOF'
通过 SSH 提交并推送到 GitHub

用法:
  git-push-github-ssh.sh [选项] [路径...]

选项:
  -m, --message MSG   提交说明（必填）
  -b, --branch NAME   推送分支（默认: master，仓库无 master 时用 main）
  -r, --remote NAME   远程名（默认: origin）
  -k, --key PATH      SSH 私钥路径（默认: ~/.ssh/id_ed25519）
  --no-add-all        不自动 git add -A，仅提交已暂存文件
  --dry-run           只打印将执行的命令，不实际提交/推送
  -h, --help          显示帮助
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!>\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31mERR>\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--message)   COMMIT_MSG="${2:-}"; shift 2 ;;
      -b|--branch)    TARGET_BRANCH="${2:-}"; shift 2 ;;
      -r|--remote)    GIT_REMOTE="${2:-}"; shift 2 ;;
      -k|--key)       SSH_KEY="${2:-}"; shift 2 ;;
      --no-add-all)   ADD_ALL=0; shift ;;
      --dry-run)      DRY_RUN=1; shift ;;
      -h|--help)      usage; exit 0 ;;
      --)             shift; PATHS+=("$@"); break ;;
      -*)             err "未知选项: $1" ;;
      *)
        if [[ -z "$COMMIT_MSG" && $# -eq 1 ]]; then
          COMMIT_MSG="$1"; shift
        else
          PATHS+=("$1"); shift
        fi
        ;;
    esac
  done
}

setup_ssh() {
  [[ -f "$SSH_KEY" ]] || err "SSH 私钥不存在: $SSH_KEY"
  log "加载 SSH 密钥: $SSH_KEY"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$(ssh-agent -s)" >/dev/null
    ssh-add "$SSH_KEY" </dev/null || err "ssh-add 失败"
  fi
  log "测试 GitHub SSH 连接"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 \
      | grep -qiE 'successfully authenticated|Hi ' || true
  fi
}

ensure_repo() {
  [[ -d "$REPO_ROOT/.git" ]] || err "不是 Git 仓库: $REPO_ROOT"
  cd "$REPO_ROOT"
  log "仓库目录: $REPO_ROOT"
}

ensure_ssh_remote() {
  local url
  url="$(git remote get-url "$GIT_REMOTE" 2>/dev/null)" || err "远程 '$GIT_REMOTE' 不存在"
  if [[ "$url" =~ ^https://github\.com/ ]]; then
    local path="${url#https://github.com/}"; path="${path%.git}"
    warn "远程为 HTTPS，切换为 SSH"
    run git remote set-url "$GIT_REMOTE" "git@github.com:${path}.git"
  else
    log "远程: $url"
  fi
}

resolve_default_branch() {
  local b="$1"
  if git show-ref --verify --quiet "refs/heads/$b" \
    || git show-ref --verify --quiet "refs/remotes/${GIT_REMOTE}/${b}"; then
    echo "$b"
    return
  fi
  if [[ "$b" == "master" ]] && git show-ref --verify --quiet refs/heads/main; then
    echo "main"
    return
  fi
  if [[ "$b" == "master" ]] && git show-ref --verify --quiet "refs/remotes/${GIT_REMOTE}/main"; then
    echo "main"
    return
  fi
  echo "$b"
}

resolve_branch() {
  if [[ -n "$TARGET_BRANCH" ]]; then
    echo "$TARGET_BRANCH"
    return
  fi
  resolve_default_branch "$DEFAULT_BRANCH"
}

checkout_target_branch() {
  local branch="$1"
  log "切换到分支: $branch"
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    run git checkout "$branch"
  elif git show-ref --verify --quiet "refs/remotes/${GIT_REMOTE}/${branch}"; then
    run git checkout -B "$branch" "${GIT_REMOTE}/${branch}"
  else
    run git checkout -b "$branch"
  fi
}

main() {
  parse_args "$@"
  command -v git >/dev/null || err "未找到 git"
  command -v ssh >/dev/null || err "未找到 ssh"
  [[ -z "$COMMIT_MSG" ]] && err "请提供提交说明: -m \"说明\""

  ensure_repo
  setup_ssh
  ensure_ssh_remote

  local branch; branch="$(resolve_branch)"
  log "目标分支: $branch（默认主分支，可用 -b 覆盖）"
  checkout_target_branch "$branch"
  run git status --short

  if [[ "$ADD_ALL" -eq 1 ]]; then
    if [[ ${#PATHS[@]} -gt 0 ]]; then run git add -- "${PATHS[@]}"; else run git add -A; fi
  fi

  if [[ "$DRY_RUN" -eq 0 ]] && git diff --cached --quiet; then
    warn "没有可提交的暂存改动"
  else
    run git commit -m "$COMMIT_MSG"
  fi

  run git push -u "$GIT_REMOTE" "$branch"
  log "完成"
}

main "$@"
