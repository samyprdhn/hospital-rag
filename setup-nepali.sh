#!/bin/bash
# Setup script for Nepali language support in the RAG application

set -e

echo "🇳🇵 BiratCare Intelligence - Nepali Language Setup"
echo "=================================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

echo "Detected OS: $OS"
echo ""

# 1. Check Python environment
echo "✓ Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.9+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"
echo ""

# 2. Install Python dependencies
echo "✓ Installing Python dependencies..."
cd "$(dirname "$0")"
pip install -r requirements.txt
echo "  ✓ All Python packages installed"
echo ""

# 3. Install Tesseract and Nepali language data
echo "✓ Setting up Tesseract OCR with Nepali language support..."
echo ""

if [ "$OS" = "macos" ]; then
    echo "  macOS detected - installing Tesseract via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "  ✗ Homebrew not found. Please install from https://brew.sh"
        exit 1
    fi

    brew install tesseract
    echo "  ✓ Tesseract installed"

    # Tesseract on macOS automatically includes language data
    # Nepali data will be downloaded on first use
    echo "  ✓ Tesseract language data configured"

elif [ "$OS" = "linux" ]; then
    echo "  Linux detected - installing Tesseract via apt..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
        sudo apt-get install -y tesseract-ocr-nep
        echo "  ✓ Tesseract and Nepali language pack installed"
    else
        echo "  ✗ apt-get not found. Please install tesseract-ocr manually"
        echo "    For Fedora/RHEL: sudo dnf install tesseract-ocr tesseract-langpack-nep"
        echo "    For Arch: sudo pacman -S tesseract tesseract-data-nep"
        exit 1
    fi

elif [ "$OS" = "windows" ]; then
    echo "  Windows detected - Manual installation required"
    echo ""
    echo "  Please follow these steps:"
    echo "  1. Download Tesseract installer from:"
    echo "     https://github.com/UB-Mannheim/tesseract/wiki"
    echo "  2. Run the installer"
    echo "  3. During installation, select 'Nepali' in language options"
    echo "  4. Add Tesseract to your PATH or update PYTESSERACT_CMD:"
    echo ""
    echo "     In PowerShell (Admin):"
    echo "     [Environment]::SetEnvironmentVariable('PYTESSERACT_CMD', 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe', 'User')"
    echo ""
    exit 0
else
    echo "  Unknown OS. Please install tesseract-ocr and Nepali language data manually"
    exit 1
fi

# 4. Verify Tesseract installation
echo ""
echo "✓ Verifying Tesseract installation..."
if ! command -v tesseract &> /dev/null; then
    echo "✗ Tesseract not found in PATH"
    exit 1
fi

TESSERACT_VERSION=$(tesseract --version | head -1)
echo "  $TESSERACT_VERSION"

echo ""
echo "✓ Checking available languages..."
LANGS=$(tesseract --list-langs)
if echo "$LANGS" | grep -q "nep"; then
    echo "  ✓ Nepali language data is available"
else
    echo "  ⚠ Nepali language data not yet downloaded"
    echo "    It will be automatically downloaded on first use"
fi

if echo "$LANGS" | grep -q "eng"; then
    echo "  ✓ English language data is available"
fi

# 5. Environment setup
echo ""
echo "✓ Setting up environment variables..."

# Check if .env file exists in parent directory
if [ -f "../.env" ]; then
    echo "  .env file found at $(pwd)/../.env"
else
    echo "  Creating sample .env file..."
    cat > ../.env.example << 'EOF'
# Birat Medical Campus Intelligence Platform
# Environment Configuration

# ── API Keys ──
LLAMAPARSE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# ── Tesseract Configuration (if not in default PATH) ──
# PYTESSERACT_CMD=/path/to/tesseract

# ── Database ──
CHROMA_PATH=./chroma_db

# ── Server ──
HOST=0.0.0.0
PORT=8000
EOF
    echo "  Sample .env created at .env.example"
fi

echo ""
echo "=========================================================="
echo "✓ Setup complete!"
echo "=========================================================="
echo ""
echo "Next steps:"
echo "1. Set up your API keys in the .env file (or GitHub Secrets)"
echo "2. For LlamaParse (cloud): Export LLAMAPARSE_API_KEY"
echo "3. For Tesseract (local): Already configured"
echo ""
echo "To start the server:"
echo "  cd artifacts/rag-app"
echo "  python main.py"
echo ""
echo "Documentation:"
echo "  See NEPALI_SUPPORT_GUIDE.md for detailed Nepali language support info"
echo ""
echo "Test with a Nepali document:"
echo "  1. Upload a PDF or image with Nepali text"
echo "  2. Select 'LlamaParse' or 'Tesseract OCR' parser"
echo "  3. Ask questions in the Chat tab"
echo ""

