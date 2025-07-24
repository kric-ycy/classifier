#!/bin/bash

# AI Text Classification - Project Structure Overview

echo "📁 AI Text Classification Project Structure"
echo "=" 50

# Main project overview
echo "🏗️  Main Project:"
tree -L 2 . 2>/dev/null || find . -maxdepth 2 -type d | sort

echo ""
echo "🧪 Test Environment (ai_net/test/):"
if [ -d "ai_net/test" ]; then
    cd ai_net/test
    ls -la *.py 2>/dev/null || echo "   No Python test files found"
    cd ../..
else
    echo "   Test directory not found"
fi

echo ""
echo "🚀 Deploy Environment (ai_net/deploy/):"
if [ -d "ai_net/deploy" ]; then
    cd ai_net/deploy
    echo "   📦 Structure:"
    tree -L 3 . 2>/dev/null || find . -maxdepth 3 -type d | sort
    echo ""
    echo "   📋 Key files:"
    ls -la *.txt *.sh *.env* 2>/dev/null || echo "      No config files found"
    cd ../..
else
    echo "   Deploy directory not found"
fi

echo ""
echo "📊 Data Directory:"
if [ -d "data" ]; then
    echo "   📁 Data structure:"
    tree data -L 2 2>/dev/null || find data -maxdepth 2 -type d | sort
    echo ""
    echo "   📈 Excel files found:"
    find data -name "*.xlsx" -o -name "*.xls" | head -5
    total_excel=$(find data -name "*.xlsx" -o -name "*.xls" | wc -l)
    echo "   Total Excel files: $total_excel"
else
    echo "   Data directory not found"
fi

echo ""
echo "🎯 Quick Start Instructions:"
echo "   Test Environment:"
echo "     cd ai_net/test"
echo "     ./start_test_env.sh"
echo ""
echo "   Production Deployment:"
echo "     cd ai_net/deploy"
echo "     ./start_production.sh"
