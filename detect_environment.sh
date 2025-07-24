#!/bin/bash

# Environment Detection and Setup Helper

echo "🔍 AI Text Classification - Environment Detection"
echo "=" 50

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if grep -q Microsoft /proc/version 2>/dev/null; then
        echo "🐧 Environment: WSL (Windows Subsystem for Linux)"
        ENV_TYPE="WSL"
    else
        echo "🐧 Environment: Native Linux"
        ENV_TYPE="LINUX"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Environment: macOS"
    ENV_TYPE="MACOS"
else
    echo "🪟 Environment: Windows"
    ENV_TYPE="WINDOWS"
fi

# Detect Python environment managers
echo ""
echo "🐍 Python Environment Managers:"

if command -v micromamba &> /dev/null; then
    echo "   ✅ Micromamba: $(micromamba --version)"
    MICROMAMBA_AVAILABLE=true
    
    echo "   📋 Micromamba environments:"
    micromamba env list | grep -v "^#" | while read env_line; do
        if [ ! -z "$env_line" ]; then
            env_name=$(echo "$env_line" | awk '{print $1}')
            if [ "$env_name" != "Name" ] && [ ! -z "$env_name" ]; then
                echo "      - $env_name"
            fi
        fi
    done
else
    echo "   ❌ Micromamba: Not found"
    MICROMAMBA_AVAILABLE=false
fi

if command -v conda &> /dev/null; then
    echo "   ✅ Conda: $(conda --version)"
    CONDA_AVAILABLE=true
    
    echo "   📋 Conda environments:"
    conda env list | grep -v "^#" | while read env_line; do
        if [ ! -z "$env_line" ]; then
            env_name=$(echo "$env_line" | awk '{print $1}')
            if [ "$env_name" != "Name" ] && [ ! -z "$env_name" ]; then
                echo "      - $env_name"
            fi
        fi
    done
else
    echo "   ❌ Conda: Not found"
    CONDA_AVAILABLE=false
fi

# Check current Python
echo ""
echo "🐍 Current Python Setup:"
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "   📦 Active environment: $CONDA_DEFAULT_ENV"
fi

if command -v python &> /dev/null; then
    echo "   🐍 Python: $(python --version)"
    echo "   📍 Location: $(which python)"
fi

if command -v python3 &> /dev/null; then
    echo "   🐍 Python3: $(python3 --version)"
    echo "   📍 Location: $(which python3)"
fi

# Environment recommendations
echo ""
echo "💡 Recommendations:"

case $ENV_TYPE in
    "WSL")
        if $MICROMAMBA_AVAILABLE; then
            echo "   ✅ WSL + Micromamba setup detected - optimal for development"
            echo "   🔧 Use: micromamba activate classifier"
        else
            echo "   💡 Consider installing micromamba for WSL environment"
        fi
        ;;
    "LINUX"|"MACOS")
        if $CONDA_AVAILABLE; then
            echo "   ✅ Native environment with conda - good for production"
            echo "   🔧 Use: conda activate classifier"
        else
            echo "   💡 Consider installing conda/miniconda for environment management"
        fi
        ;;
esac

# Show project structure expectations
echo ""
echo "📁 Project Structure Requirements:"
echo "   Current directory: $(pwd)"
echo "   Expected for test: ai_net/test/"
echo "   Expected for deploy: ai_net/deploy/"

if [ -d "ai_net/test" ] && [ -d "ai_net/deploy" ]; then
    echo "   ✅ Project structure looks good!"
    
    echo ""
    echo "🚀 Next Steps:"
    echo "   For testing:"
    echo "     cd ai_net/test"
    echo "     ./start_test_env.sh"
    echo ""
    echo "   For production:"
    echo "     cd ai_net/deploy"
    echo "     ./start_production.sh"
else
    echo "   ⚠️  Project structure incomplete"
    echo "   💡 Make sure you're in the project root directory"
fi
