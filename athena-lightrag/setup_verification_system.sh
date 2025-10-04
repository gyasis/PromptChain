#!/bin/bash

# Setup script for Claude Code Verification System
# This script sets up the complete anti-lying verification system

echo "🚀 Setting up Claude Code Verification System..."

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p .claude/commands
mkdir -p usecase-validator/src/utils
mkdir -p usecase-validator/usecases
mkdir -p usecase-validator/reports
mkdir -p usecase-validator/generated_tests
mkdir -p usecase-validator/logs

# Set up environment
echo "⚙️ Setting up environment..."
cd usecase-validator

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    cp env.example .env
    echo "📝 Created .env file from template. Please configure your settings."
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Set up the CLI
echo "🔧 Setting up CLI..."
chmod +x src/cli.py

# Run initial setup
echo "🎯 Running initial setup..."
python src/cli.py setup

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure your .env file with your API and database settings"
echo "2. Test the verification commands:"
echo "   - /verify-status 'test feature is complete'"
echo "   - /verify-api"
echo "   - /verify-real-tests"
echo "   - /verify-completion 'authentication system is complete'"
echo ""
echo "3. Create your first use case:"
echo "   cd usecase-validator"
echo "   python src/cli.py create 'My Test Case'"
echo ""
echo "4. Run use cases:"
echo "   python src/cli.py run usecases/"
echo ""
echo "🎉 You're ready to prevent Claude Code from lying!"




