#!/usr/bin/env bash
# MacBook 合盖后仍保持可被远程调用（Cursor Agent / SSH 等）
# 建议接电源使用；合盖会关屏，但系统与网络尽量保持活跃。
set -euo pipefail

STATE_DIR="${HOME}/.local/share/mac-lid-remote"
BACKUP_FILE="${STATE_DIR}/pmset-backup.env"
CAFFEINATE_PID="${STATE_DIR}/caffeinate.pid"
LAUNCH_AGENT="${HOME}/Library/LaunchAgents/com.mac-lid-remote.caffeinate.plist"
LABEL="com.mac-lid-remote.caffeinate"

die() { echo "错误: $*" >&2; exit 1; }
info() { echo ">> $*"; }

need_sudo() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    return 0
  fi
  if ! sudo -n true 2>/dev/null; then
    info "需要管理员权限以修改电源策略，请输入密码…"
  fi
  sudo -v || die "无法获取 sudo 权限"
}

# 从 pmset -g custom 读取某电源配置下的键值
_pmset_val() {
  local profile="$1" key="$2"
  local section
  case "$profile" in
    ac) section="AC Power" ;;
    batt) section="Battery Power" ;;
    *) die "未知电源配置: $profile" ;;
  esac
  pmset -g custom 2>/dev/null | awk -v sec="${section}:" -v k="${key}" '
    $0 == sec { in_sec = 1; next }
    in_sec && /^[^ \t]/ { in_sec = 0 }
    in_sec && $1 == k { print $3; exit }
  '
}

_backup_pmset() {
  mkdir -p "${STATE_DIR}"
  if [[ -f "${BACKUP_FILE}" && "${FORCE_BACKUP:-0}" != "1" ]]; then
    info "已存在备份 ${BACKUP_FILE}，跳过覆盖（恢复请用 restore 脚本；强制重备份: FORCE_BACKUP=1）"
    return 0
  fi

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  cat >"${BACKUP_FILE}" <<EOF
# mac-lid-remote 自动备份 — 请勿手改
BACKUP_CREATED_AT=${ts}
AC_sleep=$(_pmset_val ac sleep)
AC_displaysleep=$(_pmset_val ac displaysleep)
AC_disablesleep=$(_pmset_val ac disablesleep || echo 0)
AC_standby=$(_pmset_val ac standby)
AC_tcpkeepalive=$(_pmset_val ac tcpkeepalive)
AC_networkoversleep=$(_pmset_val ac networkoversleep)
AC_powernap=$(_pmset_val ac powernap)
AC_womp=$(_pmset_val ac womp)
BATT_sleep=$(_pmset_val batt sleep)
BATT_displaysleep=$(_pmset_val batt displaysleep)
BATT_disablesleep=$(_pmset_val batt disablesleep || echo 0)
BATT_standby=$(_pmset_val batt standby)
BATT_tcpkeepalive=$(_pmset_val batt tcpkeepalive)
BATT_networkoversleep=$(_pmset_val batt networkoversleep)
BATT_powernap=$(_pmset_val batt powernap)
BATT_womp=$(_pmset_val batt womp)
EOF
  info "已备份当前 pmset 到 ${BACKUP_FILE}"
}

_apply_remote_friendly_pmset() {
  need_sudo
  info "应用合盖远程友好电源策略…"
  # 接电源：合盖不进入系统睡眠（合盖仍会关屏）
  sudo pmset -c sleep 0
  sudo pmset -c disablesleep 1
  sudo pmset -c standby 0
  sudo pmset -c tcpkeepalive 1
  sudo pmset -c networkoversleep 0
  sudo pmset -c womp 1
  # 电池：尽量保持网络唤醒能力（合盖+仅电池仍可能休眠）
  sudo pmset -b tcpkeepalive 1
  sudo pmset -b networkoversleep 0
  sudo pmset -b womp 1
}

_start_caffeinate() {
  mkdir -p "${STATE_DIR}"
  if [[ -f "${CAFFEINATE_PID}" ]]; then
    local pid
    pid="$(cat "${CAFFEINATE_PID}")"
    if kill -0 "${pid}" 2>/dev/null; then
      info "caffeinate 已在运行 (PID ${pid})"
      return 0
    fi
    rm -f "${CAFFEINATE_PID}"
  fi
  # -d 显示器可睡；-i 阻止 idle sleep；-m 磁盘；-s 合盖场景；-u 用户活跃
  nohup /usr/bin/caffeinate -dims >/dev/null 2>&1 &
  echo $! >"${CAFFEINATE_PID}"
  info "已启动 caffeinate (PID $(cat "${CAFFEINATE_PID}"))"
}

_install_launch_agent() {
  mkdir -p "$(dirname "${LAUNCH_AGENT}")"
  cat >"${LAUNCH_AGENT}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-dims</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
EOF
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "${LAUNCH_AGENT}"
  launchctl enable "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  info "已安装登录自启 LaunchAgent: ${LAUNCH_AGENT}"
}

main() {
  info "Mac 合盖远程保持 — 启用"
  _backup_pmset
  _apply_remote_friendly_pmset
  _start_caffeinate
  if [[ "${INSTALL_LAUNCH_AGENT:-1}" == "1" ]]; then
    _install_launch_agent
  fi

  echo ""
  info "完成。当前有效设置："
  pmset -g custom 2>/dev/null | sed 's/^/    /' || pmset -g
  echo ""
  cat <<'NOTE'
说明：
  • 接电源合盖：系统与网络应持续在线，便于 Cursor Agent / SSH 远程调用。
  • 仅用电池合盖：macOS 仍可能休眠以省电，远程不一定稳定；远程办公请接电源。
  • 若已用 Amphetamine 等工具，可能与 caffeinate 叠加，一般无妨。
  • 恢复默认：运行同目录 mac-lid-remote-restore.sh
NOTE
}

main "$@"
