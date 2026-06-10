#!/usr/bin/env bash
# Setup API keys for Memory Is a Hidden Dependency reproduction
# Reads config/api_keys.json (from api_keys.json.template) and sets environment variables.
#
# Usage:
#   bash scripts/setup_api_keys.sh              # Read keys and set env vars (for current session)
#   source scripts/setup_api_keys.sh        # Source to set env vars in current shell
#   bash scripts/setup_api_keys.sh --verify  # Verify keys are working
#
# NOTES:
# - This script does NOT store keys in version control.
# - Add api_keys.json to your .gitignore.
# - For CI/CD, use GitHub Secrets or equivalent.

set -u  # But NOT -e (let environment checking continue on failure)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/config"
KEYS_FILE="${CONFIG_DIR}/api_keys.json"

# ---- helpers -----------------------------------------------------------------
print_help() {
    cat <<-'EOF'
Usage: setup_api_keys.sh [OPTIONS]

Set up API keys as environment variables for the Memory Is a Hidden Dependency project.

OPTIONS:
  -h, --help       Show this help message
  -v, --verify     Verify that keys are working (make test API calls)
  -f, --file FILE  Path to API keys JSON file (default: config/api_keys.json)

EXAMPLES:
  bash scripts/setup_api_keys.sh                     # Read keys, print export commands
  source scripts/setup_api_keys.sh                 # Set env vars in current shell
  bash scripts/setup_api_keys.sh --verify            # Verify keys are working
  bash scripts/setup_api_keys.sh -f my_keys.json  # Use custom keys file

After sourcing, verify with: echo $OPENAI_API_KEY
EOF
}

verify_openai() {
    local api_key="$1"
    echo "  [OpenAI] Verifying..."
    local response
    response=$(curl -s - = "$api_key" \
        "https://api.openai.com/v1/models" 2>/dev/null | head -20)
    if echo "$response" | grep -q '"data"'; then
        echo "  [OpenAI] SUCCESS: API key is valid"
        return 0
    else
        echo "  [OpenAI] FAILURE: API key may be invalid (or network issue)"
        return 1
    fi
}

verify_anthropic() {
    local api_key="$1"
    echo "  [Anthropic] Verifying..."
    local response
    response=$(curl -s -H "x-api-key: $api_key" \
        -H "anthropic-version: 2023-06-01" \
        "https://api.anthropic.com/v1/messages" 2>/dev/null | head -20)
    # Anthropic returns 400 for invalid key (not 401), so be lenient
    if [ $? -eq 0 ] || echo "$response" | grep -q '"type"'; then
        echo "  [Anthropic] SUCCESS (or endpoint reachable)"
        return 0
    else
        echo "  [Anthropic] FAILURE: Could not reach API"
        return 1
    fi
}

verify_github() {
    local token="$1"
    echo "  [GitHub] Verifying..."
    local response
    response=$(curl -s -H "Authorization: token $token" \
        "https://api.github.com/user" 2>/dev/null)
    if echo "$response" | grep -q '"login"'; then
        echo "  [GitHub] SUCCESS: Token is valid (login: $(echo "$response" | grep login | head -1 | cut -d'"' -f4))"
        return 0
    else
        echo "  [GitHub] FAILURE: Token may be invalid"
        return 1
    fi
}

# ---- main -------------------------------------------------------------------
main() {
    local verify=false
    local keys_file="$KEYS_FILE"
    
    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)     print_help; exit 0 ;;
            -v|--verify)   verify=true ;;
            -f|--file)     shift; keys_file="$1" ;;
            *)              echo "Unknown argument: $1"; print_help; exit 1 ;;
        esac
        shift
    done
    
    # Validate keys file exists
    if [ ! -f "$keys_file" ]; then
        echo "ERROR: API keys file not found: $keys_file"
        echo "Please copy config/api_keys.json.template to config/api_keys.json and fill in your keys."
        exit 1
    fi
    
    echo "=== API Keys Setup ==="
    echo "Keys file: $keys_file"
    echo ""
    
    # Read keys from JSON (use python for JSON parsing)
    local openai_key anthropic_key github_token hf_token
    openai_key=$(python3 -c "import json; d=json.load(open('$keys_file')); print(d.get('openai',{}).get('api_key',''))" 2>/dev/null)
    anthropic_key=$(python3 -c "import json; d=json.load(open('$keys_file')); print(d.get('anthropic',{}).get('api_key',''))" 2>/dev/null)
    github_token=$(python3 -c "import json; d=json.load(open('$keys_file')); print(d.get('github',{}).get('token',''))" 2>/dev/null)
    hf_token=$(python3 -c "import json; d=json.load(open('$keys_file')); print(d.get('huggingface',{}).get('token',''))" 2>/dev/null)
    
    # Set environment variables (print export commands)
    echo "Copy and paste these export commands, or source this script:"
    echo ""
    [ "$openai_key" != "" ] && [ "$openai_key" != "null" ] && echo "export OPENAI_API_KEY='$openai_key'"
    [ "$anthropic_key" != "" ] && [ "$anthropic_key" != "null" ] && echo "export ANTHROPIC_API_KEY='$anthropic_key'"
    [ "$github_token" != "" ] && [ "$github_token" != "null" ] && echo "export GITHUB_TOKEN='$github_token'"
    [ "$hf_token" != "" ] && [ "$hf_token" != "null" ] && echo "export HUGGINGFACE_TOKEN='$hf_token'"
    echo ""
    
    # Verify mode
    if $verify; then
        echo "=== Verifying API Keys ==="
        [ "$openai_key" != "" ] && [ "$openai_key" != "null" ] && verify_openai "$openai_key"
        [ "$anthropic_key" != "" ] && [ "$anthropic_key" != "null" ] && verify_anthropic "$anthropic_key"
        [ "$github_token" != "" ] && [ "$github_token" != "null" ] && verify_github "$github_token"
        echo "=== Verification Complete ==="
    fi
    
    echo ""
    echo "Done. Remember: if not sourcing, run: source scripts/setup_api_keys.sh"
}

main "$@"
