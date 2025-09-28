#!/bin/bash

# Ukrainian TTS Installation Script
# This script installs system dependencies and sets up the Python environment

set -e  # Exit on any error

echo "🚀 Installing Ukrainian TTS dependencies..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please install dependencies manually for other systems."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_error "Homebrew is not installed. Please install Homebrew first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

print_status "Installing system dependencies via Homebrew..."

# Install system dependencies
print_status "Installing SentencePiece C++ library..."
if brew list sentencepiece &> /dev/null; then
    print_success "SentencePiece already installed"
else
    brew install sentencepiece
    print_success "SentencePiece installed"
fi

# Install other system dependencies that might be needed
print_status "Installing additional system dependencies..."
brew install cmake pkg-config libsndfile

print_success "System dependencies installed"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_status "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    print_success "uv installed"
else
    print_success "uv already installed"
fi

# Set up environment variables for sentencepiece
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"

print_status "Creating virtual environment with uv..."

# Create virtual environment
uv venv --python 3.11

print_status "Activating virtual environment and installing Python dependencies..."

# Install Python dependencies
uv pip install -e .

print_success "Python dependencies installed"

# Test installation
print_status "Testing installation..."
source .venv/bin/activate

# Test critical imports
python -c "
import warnings
warnings.filterwarnings('ignore', message='Failed to import Flash Attention')
try:
    import sentencepiece
    print('✅ sentencepiece:', sentencepiece.__version__)
except ImportError as e:
    print('❌ sentencepiece import failed:', e)
    exit(1)

try:
    import espnet
    print('✅ espnet:', espnet.__version__)
except ImportError as e:
    print('❌ espnet import failed:', e)
    exit(1)

try:
    import torch
    print('✅ torch:', torch.__version__)
except ImportError as e:
    print('❌ torch import failed:', e)
    exit(1)

print('✅ All critical dependencies imported successfully!')
"

print_success "🎉 Ukrainian TTS installation completed successfully!"
print_status "To activate the environment, run: source .venv/bin/activate"
print_status "To run your TTS code, make sure to activate the environment first"

# Create a simple activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Ukrainian TTS Environment Activation Script

echo "🚀 Activating Ukrainian TTS environment..."

# Set environment variables for sentencepiece
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"

# Activate virtual environment
source .venv/bin/activate

echo "✅ Environment activated!"
echo "💡 You can now run your Ukrainian TTS code"
echo "💡 Flash Attention warnings are normal on macOS and can be ignored"
EOF

chmod +x activate_env.sh
print_success "Created activation script: ./activate_env.sh"

echo ""
print_success "Installation Summary:"
echo "  ✅ System dependencies installed (SentencePiece, CMake, pkg-config)"
echo "  ✅ Python virtual environment created"
echo "  ✅ All Python packages installed"
echo "  ✅ Installation tested and verified"
echo ""
print_status "Next steps:"
echo "  1. Run: source .venv/bin/activate"
echo "  2. Or run: ./activate_env.sh"
echo "  3. Start using your Ukrainian TTS!"
echo ""
print_warning "Note: Flash Attention warnings are normal on macOS and can be safely ignored"
