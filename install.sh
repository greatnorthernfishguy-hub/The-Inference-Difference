#!/usr/bin/env bash
#
# install.sh — One-click installer for The Inference Difference
#
# Installs the intelligent routing gateway alongside your existing
# E-T Systems modules (NeuroGraph, OpenClaw, Syl).
#
# Usage:
#   ./install.sh                # Full install + start service
#   ./install.sh --deps-only    # Only install dependencies
#   ./install.sh --no-service   # Install without systemd service
#   ./install.sh --uninstall    # Remove installation (preserves learned data)
#   ./install.sh --status       # Check installation status
#   ./install.sh --quiet        # Full install with minimal output
#
# Environment:
#   TID_INSTALL_DIR    — Override install location (default: repo directory)
#   TID_PORT           — Override API port (default: 7437)
#   TID_HOST           — Override bind host (default: 127.0.0.1)
#
# Requirements:
#   - Ubuntu 24.04 (tested), should work on 22.04+
#   - Python 3.10+
#   - ~500MB disk space (venv + dependencies)
#   - Optional: NVIDIA GPU with nvidia-smi for local model support
#   - Optional: ollama for local model serving
#
# Changelog (Grok audit response, 2026-02-19):
# - ADDED: --quiet flag for minimal output (audit: "echo spam").
# - CHANGED: Default install dir is now SCRIPT_DIR (the repo itself), not /opt/.
#   Josh's setup has ALL modules living in /home/josh/. Copying to a second
#   location (/opt/ or ~/.local/share/) causes confusion and drift. The venv,
#   data/, and logs/ directories are created inside the repo checkout. Override
#   with TID_INSTALL_DIR if you want a separate install location.
# - KEPT: Linux/systemd focus (audit: "assumes Linux"). This project targets
#   Ubuntu VPS deployment (Josh's Sylphrena infra). macOS/Windows support
#   would need launchctl/Windows Service wrappers respectively — tracked as
#   future work, not a bug. Users on those platforms can use --no-service and
#   run uvicorn directly.
# - KEPT: No git --verify-signatures (audit: "no hash verify"). This installer
#   copies from the local checkout, not from a git clone. The source is already
#   on disk. Signature verification belongs in CI/CD, not the installer.
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${TID_INSTALL_DIR:-$SCRIPT_DIR}"
VENV_DIR="$INSTALL_DIR/venv"
DATA_DIR="$INSTALL_DIR/data"
LOG_DIR="$INSTALL_DIR/logs"
CONFIG_FILE="$INSTALL_DIR/config.json"
SERVICE_NAME="inference-difference"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

PORT="${TID_PORT:-7437}"
HOST="${TID_HOST:-127.0.0.1}"

# Minimum Python version
MIN_PYTHON="3.10"

# Quiet mode (set by --quiet flag)
QUIET=false

# Skip systemd service (set when no root access)
SKIP_SERVICE=false

# ---------------------------------------------------------------------------
# Colors and logging
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { $QUIET || echo -e "${GREEN}[+]${NC} $*"; }
warn()  { $QUIET || echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*" >&2; }  # Errors always shown
header() { $QUIET || echo -e "\n${BLUE}${BOLD}$*${NC}"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

preflight() {
    header "Pre-flight checks"

    # Must be run as root (for /opt and systemd) or with sudo
    if [[ $EUID -ne 0 ]]; then
        # Check if we can sudo
        if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
            warn "Not running as root. Will use sudo for system operations."
            SUDO="sudo"
        elif command -v sudo &>/dev/null; then
            warn "Not running as root. Will use sudo for system operations."
            SUDO="sudo"
        else
            SUDO=""
            SKIP_SERVICE=true
            warn "No root/sudo access. Systemd service will be skipped."
            warn "Install dir: $INSTALL_DIR (run manually or use --no-service)."
        fi
    else
        SUDO=""
    fi

    # Check Python
    if command -v python3 &>/dev/null; then
        PYTHON="python3"
    elif command -v python &>/dev/null; then
        PYTHON="python"
    else
        error "Python 3 not found. Install with: sudo apt install python3"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if $PYTHON -c "
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    sys.exit(1)
" 2>/dev/null; then
        info "Python $PYTHON_VERSION OK"
    else
        error "Python >= $MIN_PYTHON required (found $PYTHON_VERSION)"
        exit 1
    fi

    # Check pip
    if ! $PYTHON -m pip --version &>/dev/null; then
        warn "pip not found. Installing..."
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-pip python3-venv
    fi

    # Check venv module
    if ! $PYTHON -c "import venv" &>/dev/null; then
        warn "venv module missing. Installing..."
        $SUDO apt-get install -y -qq python3-venv
    fi

    # Check for GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        info "GPU detected: $GPU_NAME (${GPU_VRAM}MB VRAM)"
        HAS_GPU=true
    else
        warn "No NVIDIA GPU detected. Local models will run on CPU."
        HAS_GPU=false
    fi

    # Check for ollama
    if command -v ollama &>/dev/null; then
        OLLAMA_MODELS=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
        info "Ollama detected with $OLLAMA_MODELS model(s)"
        HAS_OLLAMA=true
    else
        warn "Ollama not found. Install from https://ollama.ai for local models."
        HAS_OLLAMA=false
    fi

    # Check for existing NeuroGraph
    if [ -d "$HOME/.openclaw/neurograph" ] || [ -d "/home/$SUDO_USER/.openclaw/neurograph" ] 2>/dev/null; then
        info "NeuroGraph installation detected"
    fi

    info "Pre-flight checks passed"
}

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------

install_deps() {
    header "Installing dependencies"

    # Create install directory
    $SUDO mkdir -p "$INSTALL_DIR"
    $SUDO mkdir -p "$DATA_DIR"
    $SUDO mkdir -p "$LOG_DIR"

    # Set ownership to the real user (not root) — only for separate install dirs.
    # In-place installs already have the right ownership; chown -R on the repo
    # would also hit .git/ which we don't want to touch.
    REAL_USER="${SUDO_USER:-$USER}"
    REAL_HOME=$(eval echo "~$REAL_USER")
    if [ "$INSTALL_DIR" != "$SCRIPT_DIR" ]; then
        $SUDO chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
    else
        # Only chown the dirs we created
        for d in "$DATA_DIR" "$LOG_DIR"; do
            [ -d "$d" ] && $SUDO chown -R "$REAL_USER:$REAL_USER" "$d"
        done
    fi

    # Copy source files (skip if installing in-place)
    if [ "$INSTALL_DIR" != "$SCRIPT_DIR" ]; then
        info "Copying source files to $INSTALL_DIR..."
        cp "$SCRIPT_DIR/ng_lite.py" "$INSTALL_DIR/"
        cp -r "$SCRIPT_DIR/inference_difference" "$INSTALL_DIR/"
    else
        info "Installing in-place (source files already here)"
    fi

    # Create virtual environment
    info "Creating Python virtual environment..."
    $PYTHON -m venv "$VENV_DIR"

    # Install dependencies in venv
    info "Installing Python packages..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install \
        "numpy>=1.24.0" \
        "fastapi>=0.100.0" \
        "uvicorn[standard]>=0.22.0" \
        "pydantic>=2.0.0" \
        -q

    info "Dependencies installed"
}

# ---------------------------------------------------------------------------
# Hardware-aware configuration
# ---------------------------------------------------------------------------

generate_config() {
    header "Generating configuration"

    # Detect available ollama models
    OLLAMA_MODEL_LIST="[]"
    if [ "$HAS_OLLAMA" = true ]; then
        OLLAMA_MODEL_LIST=$($PYTHON -c "
import subprocess, json
result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
models = []
for line in result.stdout.strip().split('\n')[1:]:
    if line.strip():
        models.append(line.split()[0])
print(json.dumps(models))
" 2>/dev/null || echo "[]")
    fi

    # Generate config based on detected hardware
    $PYTHON -c "
import json

config = {
    'host': '$HOST',
    'port': $PORT,
    'has_gpu': $( [ \"$HAS_GPU\" = true ] && echo 'True' || echo 'False' ),
    'has_ollama': $( [ \"$HAS_OLLAMA\" = true ] && echo 'True' || echo 'False' ),
    'ollama_models': $OLLAMA_MODEL_LIST,
    'ng_lite_state_path': '$DATA_DIR/ng_lite_state.json',
    'log_file': '$LOG_DIR/inference_difference.log',
    'enable_learning': True,
    'enable_consciousness_routing': True,
    'quality_threshold': 0.7,
    'latency_budget_ms': 5000.0,
    'cost_budget_per_request': 0.10,
}

with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))
"

    info "Configuration written to $CONFIG_FILE"
}

# ---------------------------------------------------------------------------
# Systemd service
# ---------------------------------------------------------------------------

install_service() {
    header "Installing systemd service"

    REAL_USER="${SUDO_USER:-$USER}"

    $SUDO tee "$SERVICE_FILE" > /dev/null << SERVICEEOF
[Unit]
Description=The Inference Difference — Intelligent Routing Gateway
Documentation=https://github.com/greatnorthernfishguy-hub/The-Inference-Difference
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR
ExecStart=$VENV_DIR/bin/uvicorn inference_difference.app:app --host $HOST --port $PORT --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=append:$LOG_DIR/inference_difference.log
StandardError=append:$LOG_DIR/inference_difference.log

# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=$DATA_DIR $LOG_DIR
ProtectHome=read-only

[Install]
WantedBy=multi-user.target
SERVICEEOF

    $SUDO systemctl daemon-reload
    $SUDO systemctl enable "$SERVICE_NAME"
    $SUDO systemctl start "$SERVICE_NAME"

    # Wait a moment and check
    sleep 2
    if $SUDO systemctl is-active --quiet "$SERVICE_NAME"; then
        info "Service started successfully"
    else
        warn "Service may not have started. Check: journalctl -u $SERVICE_NAME"
    fi
}

# ---------------------------------------------------------------------------
# Desktop shortcut (for Ubuntu Desktop)
# ---------------------------------------------------------------------------

install_desktop_shortcut() {
    REAL_USER="${SUDO_USER:-$USER}"
    REAL_HOME=$(eval echo "~$REAL_USER")
    DESKTOP_DIR="$REAL_HOME/Desktop"

    if [ -d "$DESKTOP_DIR" ]; then
        header "Creating desktop shortcut"

        cat > "$DESKTOP_DIR/inference-difference.desktop" << DESKTOPEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=The Inference Difference
Comment=Intelligent Routing Gateway — E-T Systems
Exec=xdg-open http://$HOST:$PORT/docs
Icon=network-server
Terminal=false
Categories=Development;Science;
DESKTOPEOF
        chmod +x "$DESKTOP_DIR/inference-difference.desktop"
        chown "$REAL_USER:$REAL_USER" "$DESKTOP_DIR/inference-difference.desktop"
        info "Desktop shortcut created (opens API docs in browser)"
    fi
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

verify() {
    header "Verifying installation"
    local ok=true

    # Check files
    for f in ng_lite.py inference_difference/__init__.py inference_difference/app.py; do
        if [ ! -f "$INSTALL_DIR/$f" ]; then
            error "Missing: $INSTALL_DIR/$f"
            ok=false
        fi
    done

    # Check venv
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        error "Virtual environment missing"
        ok=false
    fi

    # Check imports
    if "$VENV_DIR/bin/python" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
from inference_difference.router import RoutingEngine
from inference_difference.classifier import classify_request
from ng_lite import NGLite
print('imports_ok')
" 2>/dev/null | grep -q "imports_ok"; then
        info "Python imports OK"
    else
        error "Python import check failed"
        ok=false
    fi

    # Check service
    if $SUDO systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        info "Service is running"

        # Check API endpoint
        if command -v curl &>/dev/null; then
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/health" 2>/dev/null || echo "000")
            if [ "$HTTP_CODE" = "200" ]; then
                info "API responding on http://$HOST:$PORT"
            else
                warn "API not yet responding (HTTP $HTTP_CODE) — may still be starting"
            fi
        fi
    else
        warn "Service is not running"
    fi

    if $ok; then
        info "Verification passed"
    else
        error "Verification failed — check errors above"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

show_status() {
    header "The Inference Difference — Status"

    echo ""
    if [ -d "$INSTALL_DIR" ]; then
        info "Installed at: $INSTALL_DIR"
    else
        error "Not installed"
        return 1
    fi

    if [ -f "$CONFIG_FILE" ]; then
        info "Config: $CONFIG_FILE"
    fi

    if $SUDO systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        info "Service: RUNNING"
        info "API: http://$HOST:$PORT"
        info "Docs: http://$HOST:$PORT/docs"

        # Show quick stats if API is up
        if command -v curl &>/dev/null; then
            HEALTH=$(curl -s "http://$HOST:$PORT/health" 2>/dev/null)
            if [ -n "$HEALTH" ]; then
                echo ""
                echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
            fi
        fi
    else
        warn "Service: STOPPED"
        echo "  Start with: sudo systemctl start $SERVICE_NAME"
    fi

    # Check for learned data
    if [ -f "$DATA_DIR/ng_lite_state.json" ]; then
        SIZE=$(du -h "$DATA_DIR/ng_lite_state.json" | cut -f1)
        info "Learned data: $DATA_DIR/ng_lite_state.json ($SIZE)"
    else
        info "No learned data yet (will accumulate with use)"
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

uninstall() {
    header "Uninstalling The Inference Difference"

    warn "This will remove the service and installation files."
    warn "Learned data in $DATA_DIR will be PRESERVED."
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Cancelled."
        return
    fi

    # Stop and disable service
    if [ -f "$SERVICE_FILE" ]; then
        $SUDO systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        $SUDO systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        $SUDO rm -f "$SERVICE_FILE"
        $SUDO systemctl daemon-reload
        info "Service removed"
    fi

    # Remove installation (preserve data)
    if [ -d "$INSTALL_DIR" ]; then
        # Back up learned data
        if [ -f "$DATA_DIR/ng_lite_state.json" ]; then
            BACKUP="$HOME/inference_difference_learned_backup.json"
            cp "$DATA_DIR/ng_lite_state.json" "$BACKUP"
            info "Learned data backed up to $BACKUP"
        fi

        $SUDO rm -rf "$VENV_DIR"
        $SUDO rm -f "$INSTALL_DIR/ng_lite.py"
        $SUDO rm -rf "$INSTALL_DIR/inference_difference"
        $SUDO rm -f "$CONFIG_FILE"
        info "Installation files removed"
        info "Data directory preserved: $DATA_DIR"
    fi

    # Remove desktop shortcut
    REAL_USER="${SUDO_USER:-$USER}"
    REAL_HOME=$(eval echo "~$REAL_USER")
    rm -f "$REAL_HOME/Desktop/inference-difference.desktop" 2>/dev/null

    info "Uninstall complete"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-}" in
    --deps-only)
        preflight
        install_deps
        ;;
    --no-service)
        preflight
        install_deps
        generate_config
        verify
        echo ""
        info "Installed without service. Start manually with:"
        echo "  cd $INSTALL_DIR && $VENV_DIR/bin/uvicorn inference_difference.app:app --host $HOST --port $PORT"
        ;;
    --quiet)
        QUIET=true
        preflight
        install_deps
        generate_config
        if [ "$SKIP_SERVICE" != true ]; then
            install_service
        fi
        verify
        echo "The Inference Difference installed: http://$HOST:$PORT"
        ;;
    --uninstall)
        SUDO="${SUDO:-}"
        [[ $EUID -ne 0 ]] && SUDO="sudo"
        uninstall
        ;;
    --status)
        SUDO="${SUDO:-}"
        [[ $EUID -ne 0 ]] && SUDO="sudo"
        show_status
        ;;
    --help|-h)
        echo "The Inference Difference — One-Click Installer"
        echo ""
        echo "Usage: ./install.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  (none)         Full install + start systemd service"
        echo "  --deps-only    Only install dependencies"
        echo "  --no-service   Install without systemd service"
        echo "  --quiet        Full install with minimal output"
        echo "  --uninstall    Remove installation (preserves learned data)"
        echo "  --status       Check installation status"
        echo "  --help         Show this help"
        echo ""
        echo "Environment:"
        echo "  TID_INSTALL_DIR   Install location (default: repo directory)"
        echo "  TID_PORT          API port (default: 7437)"
        echo "  TID_HOST          Bind host (default: 127.0.0.1)"
        ;;
    *)
        echo ""
        echo -e "${BOLD}========================================${NC}"
        echo -e "${BOLD} The Inference Difference${NC}"
        echo -e "${BOLD} Intelligent Routing Gateway${NC}"
        echo -e "${BOLD} E-T Systems / NeuroGraph Foundation${NC}"
        echo -e "${BOLD}========================================${NC}"
        echo ""

        preflight
        echo ""
        install_deps
        echo ""
        generate_config
        echo ""
        if [ "$SKIP_SERVICE" != true ]; then
            install_service
            echo ""
            install_desktop_shortcut
        else
            warn "Skipping systemd service (no root access)."
            info "Start manually with:"
            echo "  cd $INSTALL_DIR && $VENV_DIR/bin/uvicorn inference_difference.app:app --host $HOST --port $PORT"
        fi
        echo ""
        verify
        echo ""

        header "Installation complete!"
        echo ""
        echo -e "  ${BOLD}API:${NC}    http://$HOST:$PORT"
        echo -e "  ${BOLD}Docs:${NC}   http://$HOST:$PORT/docs"
        echo -e "  ${BOLD}Health:${NC} http://$HOST:$PORT/health"
        echo -e "  ${BOLD}Stats:${NC}  http://$HOST:$PORT/stats"
        echo ""
        echo "  Service management:"
        echo "    sudo systemctl status $SERVICE_NAME"
        echo "    sudo systemctl restart $SERVICE_NAME"
        echo "    journalctl -u $SERVICE_NAME -f"
        echo ""
        echo "  Quick test:"
        echo "    curl -X POST http://$HOST:$PORT/route \\"
        echo "      -H 'Content-Type: application/json' \\"
        echo "      -d '{\"message\": \"Write a Python function to sort a list\"}'"
        echo ""
        ;;
esac
