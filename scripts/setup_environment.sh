#!/usr/bin/env bash
# ==============================================================================
# setup_environment.sh - Environment Setup for "Memory Is a Hidden Dependency"
#                        Paper Reproduction (Stage 8, Part 3)
#
# This script checks system dependencies, installs Python dependencies,
# optionally installs the Lean compiler, sets environment variables,
# and creates necessary directories for the reproduction workflow.
#
# Usage:
#   ./setup_environment.sh           # Full setup
#   ./setup_environment.sh --check-only  # Only check dependencies, do not install
#   ./setup_environment.sh --with-lean   # Install Lean compiler (via elan)
# ==============================================================================

# ==============================================================================
# Section 0: Configuration and Constants
# ==============================================================================

# Script directory (resolve to absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root directory (assuming scripts/ is under code/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Key directories to create
DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/results"
CACHE_DIR="${PROJECT_ROOT}/cache"

# Color codes for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Default flags
CHECK_ONLY=false
WITH_LEAN=false

# ==============================================================================
# Section 0.5: Script Integrity Assertions
# ==============================================================================

# Assert that script directory resolution worked
# This is a programming invariant, not an environment check
[ -n "${SCRIPT_DIR}" ] || { echo "ASSERTION FAILED: SCRIPT_DIR is empty"; exit 1; }
[ -d "${SCRIPT_DIR}" ] || { echo "ASSERTION FAILED: SCRIPT_DIR is not a directory: ${SCRIPT_DIR}"; exit 1; }

# Assert that project root resolution worked
[ -n "${PROJECT_ROOT}" ] || { echo "ASSERTION FAILED: PROJECT_ROOT is empty"; exit 1; }
[ -d "${PROJECT_ROOT}" ] || { echo "ASSERTION FAILED: PROJECT_ROOT is not a directory: ${PROJECT_ROOT}"; exit 1; }

# Assert that required directories are properly defined
[ -n "${DATA_DIR}" ] || { echo "ASSERTION FAILED: DATA_DIR is empty"; exit 1; }
[ -n "${RESULTS_DIR}" ] || { echo "ASSERTION FAILED: RESULTS_DIR is empty"; exit 1; }
[ -n "${CACHE_DIR}" ] || { echo "ASSERTION FAILED: CACHE_DIR is empty"; exit 1; }

# ==============================================================================
# Section 1: Utility Functions
# ==============================================================================

# Print informational message
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Print success message
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Print warning message
log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Print error message
log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Print section header
print_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo ""
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Compare version numbers
# Returns 0 if $1 >= $2, 1 otherwise
version_gte() {
    local version=$1
    local required=$2

    # Split versions into arrays
    IFS='.' read -ra VERSION_PARTS <<< "$version"
    IFS='.' read -ra REQUIRED_PARTS <<< "$required"

    # Compare each part
    local i
    for (( i=0; i<${#REQUIRED_PARTS[@]}; i++ )); do
        local v_part=${VERSION_PARTS[$i]:-0}
        local r_part=${REQUIRED_PARTS[$i]}

        # Remove any non-numeric suffix (e.g., "10a" -> "10")
        v_part=$(echo "$v_part" | sed 's/[^0-9].*//')
        r_part=$(echo "$r_part" | sed 's/[^0-9].*//')

        # Compare
        if [ "$v_part" -gt "$r_part" ] 2>/dev/null; then
            return 0
        elif [ "$v_part" -lt "$r_part" ] 2>/dev/null; then
            return 1
        fi
    done

    return 0
}

# ==============================================================================
# Section 2: Argument Parsing
# ==============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --with-lean)
                WITH_LEAN=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment setup script for 'Memory Is a Hidden Dependency' paper reproduction."
    echo ""
    echo "Options:"
    echo "  --check-only    Only check dependencies, do not install anything"
    echo "  --with-lean     Install Lean compiler (via elan)"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Full setup (check + install)"
    echo "  $0 --check-only     # Only check system dependencies"
    echo "  $0 --with-lean      # Full setup including Lean installation"
}

# ==============================================================================
# Section 3: System Dependency Checks
# ==============================================================================

check_python() {
    print_section "Checking Python"

    if ! command_exists python3; then
        log_error "Python 3 is not installed or not in PATH"
        log_info "Please install Python 3.10 or higher"
        return 1
    fi

    local python_version
    python_version=$(python3 --version 2>&1 | sed 's/Python //')
    local major minor
    IFS='.' read -r major minor _ <<< "$python_version"

    log_info "Found Python $python_version"

    if [ "$major" -ne 3 ] || [ "$minor" -lt 10 ]; then
        log_error "Python version $python_version is too old. Required: 3.10+"
        return 1
    fi

    log_success "Python $python_version meets requirement (>= 3.10)"
    return 0
}

check_git() {
    print_section "Checking Git"

    if ! command_exists git; then
        log_error "Git is not installed or not in PATH"
        log_info "Please install Git: https://git-scm.com/"
        return 1
    fi

    local git_version
    git_version=$(git --version 2>&1 | sed 's/git version //')
    log_info "Found Git $git_version"
    log_success "Git is available"
    return 0
}

check_curl() {
    print_section "Checking Curl"

    if command_exists curl; then
        local curl_version
        curl_version=$(curl --version 2>&1 | head -1 | sed 's/curl //' | awk '{print $1}')
        log_info "Found curl $curl_version"
        log_success "Curl is available"
        return 0
    elif command_exists wget; then
        local wget_version
        wget_version=$(wget --version 2>&1 | head -1 | awk '{print $3}')
        log_info "Found wget $wget_version (curl not found, wget available as alternative)"
        log_success "Wget is available (can be used instead of curl)"
        return 0
    else
        log_error "Neither curl nor wget is installed"
        log_info "Please install curl or wget"
        return 1
    fi
}

check_pip() {
    print_section "Checking Pip"

    if command_exists pip3; then
        local pip_version
        pip_version=$(pip3 --version 2>&1 | awk '{print $2}')
        log_info "Found pip $pip_version"
        log_success "Pip is available"
        return 0
    elif python3 -m pip --version >/dev/null 2>&1; then
        log_info "Found pip (via python3 -m pip)"
        log_success "Pip is available"
        return 0
    else
        log_error "Pip is not installed"
        log_info "Please install pip: https://pip.pypa.io/"
        return 1
    fi
}

check_system_dependencies() {
    print_section "System Dependency Checks"

    local all_passed=true

    check_python
    [ $? -ne 0 ] && all_passed=false

    check_git
    [ $? -ne 0 ] && all_passed=false

    check_curl
    [ $? -ne 0 ] && all_passed=false

    check_pip
    [ $? -ne 0 ] && all_passed=false

    echo ""
    if [ "$all_passed" = true ]; then
        log_success "All system dependencies are satisfied"
        return 0
    else
        log_error "Some system dependencies are missing"
        return 1
    fi
}

# ==============================================================================
# Section 4: Python Dependency Installation
# ==============================================================================

install_python_dependencies() {
    print_section "Installing Python Dependencies"

    local requirements_file="${PROJECT_ROOT}/code/requirements.txt"

    if [ ! -f "$requirements_file" ]; then
        log_warn "Requirements file not found: $requirements_file"
        log_info "Skipping Python dependency installation"
        return 0
    fi

    log_info "Requirements file: $requirements_file"
    log_info "Installing dependencies..."

    # Try pip3 first, then python3 -m pip
    local pip_cmd
    if command_exists pip3; then
        pip_cmd="pip3"
    else
        pip_cmd="python3 -m pip"
    fi

    # Install dependencies
    $pip_cmd install -r "$requirements_file" 2>&1 | tee "${RESULTS_DIR}/pip_install.log"
    local install_status=$?

    if [ $install_status -ne 0 ]; then
        log_error "Failed to install Python dependencies (exit code: $install_status)"
        log_info "Check ${RESULTS_DIR}/pip_install.log for details"
        return 1
    fi

    log_success "Python dependencies installed successfully"
    return 0
}

# ==============================================================================
# Section 5: Lean Compiler Installation
# ==============================================================================

check_lean() {
    print_section "Checking Lean Compiler"

    if command_exists lake; then
        local lean_version
        lean_version=$(lake --version 2>&1 | head -1)
        log_info "Found Lean: $lean_version"
        log_success "Lean compiler is available"
        return 0
    elif command_exists elan; then
        log_info "Found elan (Lean version manager) but Lean not yet installed"
        log_info "Run with --with-lean to install Lean"
        return 0
    else
        log_warn "Lean compiler is not installed"
        log_info "Run with --with-lean to install Lean via elan"
        return 1
    fi
}

install_lean() {
    print_section "Installing Lean Compiler"

    # Check if already installed
    if command_exists lake; then
        local lean_version
        lean_version=$(lake --version 2>&1 | head -1)
        log_info "Lean already installed: $lean_version"
        log_success "Skipping Lean installation"
        return 0
    fi

    # Install elan (Lean version manager)
    log_info "Installing elan (Lean version manager)..."

    if ! command_exists elan; then
        log_info "Downloading and installing elan..."
        curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y 2>&1 | tee "${RESULTS_DIR}/elan_install.log"
        local elan_status=$?

        if [ $elan_status -ne 0 ]; then
            log_error "Failed to install elan (exit code: $elan_status)"
            log_info "Check ${RESULTS_DIR}/elan_install.log for details"
            return 1
        fi

        # Source elan environment
        if [ -f "${HOME}/.elan/env" ]; then
            source "${HOME}/.elan/env"
        fi

        log_success "Elan installed successfully"
    else
        log_info "Elan already installed"
    fi

    # Install Lean via elan
    log_info "Installing Lean 4 stable..."
    elan toolchain install stable 2>&1 | tee "${RESULTS_DIR}/lean_install.log"
    local lean_status=$?

    if [ $lean_status -ne 0 ]; then
        log_error "Failed to install Lean (exit code: $lean_status)"
        log_info "Check ${RESULTS_DIR}/lean_install.log for details"
        return 1
    fi

    # Verify installation
    if command_exists lake; then
        local lean_version
        lean_version=$(lake --version 2>&1 | head -1)
        log_success "Lean installed successfully: $lean_version"
    else
        log_warn "Lean installed but 'lake' command not found in PATH"
        log_info "You may need to restart your shell or run: source ~/.elan/env"
    fi

    return 0
}

# ==============================================================================
# Section 6: Environment Variable Setup
# ==============================================================================

setup_environment_variables() {
    print_section "Setting Environment Variables"

    local env_file="${PROJECT_ROOT}/.env"

    log_info "Environment file: $env_file"

    # Create .env file if it doesn't exist
    if [ ! -f "$env_file" ]; then
        log_info "Creating .env file..."

        cat > "$env_file" << 'EOF'
# Environment variables for "Memory Is a Hidden Dependency" reproduction
# Copy this file to .env and fill in your values

# GitHub API token (required for collecting GitHub traces)
# Get your token at: https://github.com/settings/tokens
# GITHUB_TOKEN=ghp_your_token_here

# Hugging Face token (required for accessing some datasets)
# HUGGING_FACE_HUB_TOKEN=hf_your_token_here

# OpenAI API key (required for some LLM-based analysis)
# OPENAI_API_KEY=sk_your_key_here

# Anthropic API key (required for Claude-based analysis)
# ANTHROPIC_API_KEY=your_key_here

# Data directory
DATA_DIR=./data

# Results directory
RESULTS_DIR=./results

# Cache directory
CACHE_DIR=./cache

# Number of parallel workers for data collection
NUM_WORKERS=4

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
EOF

        log_success ".env file created at $env_file"
        log_warn "Please edit $env_file and fill in your API tokens"
    else
        log_info ".env file already exists"
    fi

    # Check if GITHUB_TOKEN is set
    if [ -z "${GITHUB_TOKEN:-}" ]; then
        log_warn "GITHUB_TOKEN is not set"
        log_info "Some scripts may require GITHUB_TOKEN for GitHub API access"
        log_info "You can set it in $env_file or export it in your shell"
    else
        log_success "GITHUB_TOKEN is set"
    fi

    # Source .env file if it exists
    if [ -f "$env_file" ]; then
        log_info "Sourcing $env_file..."
        set -a
        source "$env_file"
        set +a
    fi

    return 0
}

# ==============================================================================
# Section 7: Directory Creation
# ==============================================================================

create_directories() {
    print_section "Creating Required Directories"

    local dirs=(
        "$DATA_DIR"
        "$RESULTS_DIR"
        "$CACHE_DIR"
        "${RESULTS_DIR}/figures"
        "${RESULTS_DIR}/tables"
        "${RESULTS_DIR}/logs"
    )

    local all_created=true

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
            if [ $? -ne 0 ]; then
                log_error "Failed to create directory: $dir"
                all_created=false
            fi
        else
            log_info "Directory already exists: $dir"
        fi
    done

    echo ""
    if [ "$all_created" = true ]; then
        log_success "All directories created successfully"
        return 0
    else
        log_error "Some directories could not be created"
        return 1
    fi
}

# ==============================================================================
# Section 8: Main Script Logic
# ==============================================================================

main() {
    echo ""
    echo -e "${BLUE}=============================================================================="
    echo "  Environment Setup for 'Memory Is a Hidden Dependency' Paper Reproduction"
    echo "  Stage 8, Part 3: Environment Setup"
    echo "==============================================================================${NC}"
    echo ""

    # Parse command line arguments
    parse_arguments "$@"

    log_info "Project root: $PROJECT_ROOT"
    log_info "Check-only mode: $CHECK_ONLY"
    log_info "Install Lean: $WITH_LEAN"
    echo ""

    # Step 1: Check system dependencies
    check_system_dependencies
    local deps_status=$?

    if [ $deps_status -ne 0 ]; then
        log_error "System dependency check failed"
        log_info "Please install missing dependencies and re-run this script"
        exit 1
    fi

    # If check-only mode, exit here
    if [ "$CHECK_ONLY" = true ]; then
        log_info "Check-only mode: skipping installation steps"
        log_success "All checks passed. No installations performed."
        exit 0
    fi

    # Step 2: Install Python dependencies
    install_python_dependencies
    local pip_status=$?

    if [ $pip_status -ne 0 ]; then
        log_error "Python dependency installation failed"
        log_info "You may try installing manually: pip install -r code/requirements.txt"
        # Continue anyway - some dependencies may be optional
    fi

    # Step 3: Install Lean (if requested)
    if [ "$WITH_LEAN" = true ]; then
        install_lean
        local lean_status=$?

        if [ $lean_status -ne 0 ]; then
            log_error "Lean installation failed"
            log_info "You may try installing manually: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y"
            # Continue anyway - Lean is optional for basic reproduction
        fi
    else
        # Just check if Lean is available
        check_lean
    fi

    # Step 4: Setup environment variables
    setup_environment_variables
    local env_status=$?

    if [ $env_status -ne 0 ]; then
        log_warn "Environment variable setup had issues"
        # Continue anyway - not critical
    fi

    # Step 5: Create required directories
    create_directories
    local dir_status=$?

    if [ $dir_status -ne 0 ]; then
        log_error "Directory creation failed"
        exit 1
    fi

    # Final summary
    print_section "Setup Summary"

    echo "Setup completed at: $(date)"
    echo ""
    echo "Project root: $PROJECT_ROOT"
    echo "Data directory: $DATA_DIR"
    echo "Results directory: $RESULTS_DIR"
    echo "Cache directory: $CACHE_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Edit ${PROJECT_ROOT}/.env and fill in your API tokens"
    echo "  2. Run the data collection scripts: python code/scripts/collect_github_traces.py"
    echo "  3. Run the analysis scripts: python code/scripts/summarize_results.py"
    echo ""
    log_success "Environment setup completed successfully!"
}

# Run main function with all arguments
main "$@"
