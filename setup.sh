#!/usr/bin/env bash
# Quick setup script for new users

set -euo pipefail

echo "🚀 Setting up Trading Bot System..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Install dependencies
echo "📥 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your API keys!"
    chmod 600 .env
    echo "✓ .env file created (permissions set to 600)"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
mkdir -p logs
mkdir -p state
mkdir -p backups
echo "✓ Created necessary directories (logs, state, backups)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys:"
echo "   - TELEGRAM_BOT_TOKEN"
echo "   - TELEGRAM_CHAT_ID"
echo "   - MEXC_API_KEY (if using orb_bot)"
echo "   - MEXC_API_SECRET (if using orb_bot)"
echo ""
echo "2. Validate your configuration:"
echo "   python validate_config.py"
echo ""
echo "3. Test the system:"
echo "   python run_bots.py --status"
echo ""
echo "4. Start all bots:"
echo "   ./start_all_bots.sh"
