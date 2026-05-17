#!/usr/bin/env bash
# 恢复 mac-lid-remote-enable.sh 修改前的电源与 caffeinate 设置
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
    info "需要管理员权限以恢复电源策略，请输入密码…"
  fi
  sudo -v || die "无法获取 sudo 权限"
}

_stop_caffeinate() {
  if [[ -f "${CAFFEINATE_PID}" ]]; then
    local pid
    pid="$(cat "${CAFFEINATE_PID}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      info "已停止 caffeinate (PID ${pid})"
    fi
    rm -f "${CAFFEINATE_PID}"
  fi
  pkill -f "caffeinate -dims" 2>/dev/null || true
}

_uninstall_launch_agent() {
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  if [[ -f "${LAUNCH_AGENT}" ]]; then
    rm -f "${LAUNCH_AGENT}"
    info "已移除 LaunchAgent"
  fi
}

_restore_pmset() {
  [[ -f "${BACKUP_FILE}" ]] || die "未找到备份 ${BACKUP_FILE}（是否尚未运行过 enable 脚本？）"
  # shellcheck source=/dev/null
  source "${BACKUP_FILE}"

  need_sudo
  info "从备份恢复 pmset（创建于 ${BACKUP_CREATED_AT:-未知}）…"

  restore_ac() {
    sudo pmset -c sleep "${AC_sleep}"
    sudo pmset -c displaysleep "${AC_displaysleep}"
    sudo pmset -c disablesleep "${AC_disablesleep:-0}"
    sudo pmset -c standby "${AC_standby}"
    sudo pmset -c tcpkeepalive "${AC_tcpkeepalive}"
    sudo pmset -c networkoversleep "${AC_networkoversleep}"
    sudo pmset -c powernap "${AC_powernap}"
    sudo pmset -c womp "${AC_womp}"
  }

  restore_batt() {
    sudo pmset -b sleep "${BATT_sleep}"
    sudo pmset -b displaysleep "${BATT_displaysleep}"
    sudo pmset -b disablesleep "${BATT_disablesleep:-0}"
    sudo pmset -b standby "${BATT_standby}"
    sudo pmset -b tcpkeepalive "${BATT_tcpkeepalive}"
    sudo pmset -b networkoversleep "${BATT_networkoversleep}"
    sudo pmset -b powernap "${BATT_powernap}"
    sudo pmset -b womp "${BATT_womp}"
  }

  restore_ac
  restore_batt
  info "pmset 已恢复"
}

main() {
  info "Mac 合盖远程保持 — 恢复默认"
  _stop_caffeinate
  _uninstall_launch_agent
  _restore_pmset

  if [[ "${KEEP_BACKUP:-0}" != "1" ]]; then
    rm -f "${BACKUP_FILE}"
    info "已删除备份文件（保留备份: KEEP_BACKUP=1）"
  fi

  echo ""
  info "完成。当前有效设置："
  pmset -g custom 2>/dev/null | sed 's/^/    /' || pmset -g
  echo ""
  info "若仍显示 sleep prevented by …，可能是 Amphetamine / Cursor 等应用在阻止睡眠，属正常现象。"
}

main "$@"
