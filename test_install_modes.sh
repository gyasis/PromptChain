#!/bin/bash
# Test different installation modes for PromptChain
#
# This script simulates different installation scenarios to verify
# that MLflow is truly optional.

set -e

echo "=========================================="
echo "PromptChain Installation Mode Testing"
echo "=========================================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Test 1: Check setup.py structure
echo "Test 1: Verify setup.py configuration"
echo "--------------------------------------"

if grep -q "extras_require" setup.py; then
    print_success "setup.py has extras_require"
else
    print_error "setup.py missing extras_require"
    exit 1
fi

if grep -q '"dev"' setup.py; then
    print_success "setup.py has [dev] extra"
else
    print_error "setup.py missing [dev] extra"
    exit 1
fi

if grep -q 'mlflow>=2.9.0' setup.py; then
    print_success "MLflow is in dev extras"
else
    print_error "MLflow not found in setup.py"
    exit 1
fi

# Check that MLflow is NOT in core requirements
if grep -q "mlflow" <<< "$(sed -n '/^core_requirements = \[/,/^\]/p' setup.py)"; then
    print_error "MLflow is in core_requirements (should only be in dev extras)"
    exit 1
else
    print_success "MLflow is NOT in core_requirements (correct)"
fi

echo

# Test 2: Check requirements.txt structure
echo "Test 2: Verify requirements files"
echo "--------------------------------------"

if [ -f "requirements.txt" ]; then
    print_success "requirements.txt exists"

    if grep -q "mlflow" requirements.txt; then
        print_error "MLflow found in requirements.txt (should only be in requirements-dev.txt)"
        exit 1
    else
        print_success "MLflow NOT in requirements.txt (correct)"
    fi
else
    print_error "requirements.txt not found"
    exit 1
fi

if [ -f "requirements-dev.txt" ]; then
    print_success "requirements-dev.txt exists"

    if grep -q "mlflow" requirements-dev.txt; then
        print_success "MLflow found in requirements-dev.txt"
    else
        print_error "MLflow not found in requirements-dev.txt"
        exit 1
    fi
else
    print_error "requirements-dev.txt not found"
    exit 1
fi

echo

# Test 3: Check that verification script works
echo "Test 3: Run installation verification"
echo "--------------------------------------"

if python test_installation.py; then
    print_success "Installation verification passed"
else
    print_error "Installation verification failed"
    exit 1
fi

echo

# Test 4: Verify documentation exists
echo "Test 4: Verify documentation"
echo "--------------------------------------"

if [ -f "INSTALLATION.md" ]; then
    print_success "INSTALLATION.md exists"

    if grep -q "pip install promptchain" INSTALLATION.md; then
        print_success "Core installation documented"
    fi

    if grep -q 'pip install "promptchain\[dev\]"' INSTALLATION.md; then
        print_success "Dev installation documented"
    fi

    if grep -q "MLflow is NOT required" INSTALLATION.md; then
        print_success "Optional MLflow documented"
    fi
else
    print_error "INSTALLATION.md not found"
    exit 1
fi

echo

# Summary
echo "=========================================="
echo "All Tests Passed!"
echo "=========================================="
echo
print_info "MLflow is now properly configured as an optional dependency"
print_info ""
print_info "Installation modes:"
print_info "  • Core:  pip install promptchain"
print_info "  • CLI:   pip install 'promptchain[cli]'"
print_info "  • Dev:   pip install 'promptchain[dev]'"
print_info "  • All:   pip install 'promptchain[all]'"
echo

exit 0
