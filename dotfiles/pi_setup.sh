#!/usr/bin/env bash
# ── Raspberry Pi Lab Node: First-Login Hardening ──────────────────────────────
# Run as: bash pi_setup.sh <your-ssh-public-key>
# Example: bash pi_setup.sh "$(cat ~/.ssh/id_ed25519.pub)"
#
# What this does:
#   1. Installs vim, tmux, git, python3, cargo (Rust)
#   2. Copies your SSH public key → authorized_keys
#   3. Disables password login (key-only SSH)
#   4. Hardens sshd_config: no root login, no empty passwords
#   5. Copies dotfiles (vimrc, tmux.conf)
#   6. Sets up the Python venv for this project
#   7. Optionally installs jupyter lab as a systemd service

set -euo pipefail
PUBKEY="${1:-}"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; CYN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYN}[info]${NC}  $*"; }
ok()    { echo -e "${GRN}[ok]${NC}    $*"; }
warn()  { echo -e "${YLW}[warn]${NC}  $*"; }
die()   { echo -e "${RED}[err]${NC}   $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && die "Run as root: sudo bash $0 \"<pubkey>\""
[[ -z "$PUBKEY" ]] && die "Pass your SSH public key as arg 1"

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    vim tmux git curl build-essential \
    python3 python3-pip python3-venv \
    sqlite3 ripgrep fd-find \
    ufw fail2ban
ok "Packages installed"

# Install Rust (cargo) if not present
if ! command -v cargo &>/dev/null; then
    info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    source "$HOME/.cargo/env"
    ok "Rust $(rustc --version)"
fi

# ── 2. SSH key auth ───────────────────────────────────────────────────────────
LAB_USER="${SUDO_USER:-pi}"
HOME_DIR=$(eval echo "~$LAB_USER")
SSH_DIR="$HOME_DIR/.ssh"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

AUTHKEYS="$SSH_DIR/authorized_keys"
if grep -qF "$PUBKEY" "$AUTHKEYS" 2>/dev/null; then
    warn "Key already in authorized_keys"
else
    echo "$PUBKEY" >> "$AUTHKEYS"
    ok "SSH public key added to $AUTHKEYS"
fi
chmod 600 "$AUTHKEYS"
chown -R "$LAB_USER:$LAB_USER" "$SSH_DIR"

# ── 3. Harden sshd_config ─────────────────────────────────────────────────────
SSHD="/etc/ssh/sshd_config"
info "Hardening sshd..."

# Backup once
[[ -f "$SSHD.orig" ]] || cp "$SSHD" "$SSHD.orig"

set_sshd() {
    local key="$1" val="$2"
    if grep -qE "^#?${key}" "$SSHD"; then
        sed -i "s|^#\?${key}.*|${key} ${val}|" "$SSHD"
    else
        echo "${key} ${val}" >> "$SSHD"
    fi
}

set_sshd "PasswordAuthentication"    "no"
set_sshd "PermitRootLogin"           "no"
set_sshd "PermitEmptyPasswords"      "no"
set_sshd "ChallengeResponseAuthentication" "no"
set_sshd "UsePAM"                    "yes"
set_sshd "X11Forwarding"             "no"
set_sshd "PrintMotd"                 "no"
set_sshd "MaxAuthTries"              "3"
set_sshd "ClientAliveInterval"       "300"
set_sshd "ClientAliveCountMax"       "2"
set_sshd "AllowUsers"                "$LAB_USER"

systemctl restart ssh
ok "sshd hardened: key-only, no root, no password auth"

# ── 4. Firewall ───────────────────────────────────────────────────────────────
info "Configuring UFW firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8888/tcp comment "jupyter lab"
ufw --force enable
ok "UFW active: SSH + Jupyter only"

# ── 5. Dotfiles ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/vimrc" ]]; then
    cp "$SCRIPT_DIR/vimrc" "$HOME_DIR/.vimrc"
    chown "$LAB_USER:$LAB_USER" "$HOME_DIR/.vimrc"
    ok "vimrc installed"
fi

if [[ -f "$SCRIPT_DIR/tmux.conf" ]]; then
    cp "$SCRIPT_DIR/tmux.conf" "$HOME_DIR/.tmux.conf"
    chown "$LAB_USER:$LAB_USER" "$HOME_DIR/.tmux.conf"
    ok "tmux.conf installed"
fi

# ── 6. Python venv for the project ───────────────────────────────────────────
PROJ_DIR="$HOME_DIR/Dispersion-Assisted-GS-Phase-Recovery"
if [[ -d "$PROJ_DIR" ]]; then
    info "Setting up Python venv in project..."
    sudo -u "$LAB_USER" python3 -m venv "$PROJ_DIR/.venv"
    sudo -u "$LAB_USER" "$PROJ_DIR/.venv/bin/pip" install -q --upgrade pip
    if [[ -f "$PROJ_DIR/requirements.txt" ]]; then
        sudo -u "$LAB_USER" "$PROJ_DIR/.venv/bin/pip" install -q -r "$PROJ_DIR/requirements.txt"
        ok "Project dependencies installed in .venv"
    fi
fi

# ── 7. Jupyter Lab systemd service (optional) ─────────────────────────────────
read -rp "Install Jupyter Lab as systemd service? [y/N] " ans
if [[ "${ans,,}" == "y" ]]; then
    VENV_PYTHON="$PROJ_DIR/.venv/bin/python"
    [[ -f "$VENV_PYTHON" ]] || VENV_PYTHON="$(which python3)"

    cat > /etc/systemd/system/jupyterlab.service <<EOF
[Unit]
Description=Jupyter Lab (Jalali Lab node)
After=network.target

[Service]
Type=simple
User=$LAB_USER
WorkingDirectory=$PROJ_DIR
ExecStart=$VENV_PYTHON -m jupyter lab \\
    --no-browser \\
    --ip=0.0.0.0 \\
    --port=8888 \\
    --NotebookApp.token='' \\
    --NotebookApp.password=''
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable jupyterlab
    systemctl start  jupyterlab
    ok "Jupyter Lab service enabled (port 8888)"
    warn "No token set — only access via SSH tunnel:"
    echo "  ssh -L 8888:localhost:8888 $LAB_USER@<pi-ip>"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}━━━ Setup complete ━━━${NC}"
echo "SSH in with:  ssh $LAB_USER@<pi-ip>"
echo "Tunnel JLab:  ssh -L 8888:localhost:8888 $LAB_USER@<pi-ip>"
echo "Then open:    http://localhost:8888"
