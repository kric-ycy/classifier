#!/bin/bash

# AI Text Classification - Test Environment

echo "🧪 AI Text Classification - Test Environment"
echo "=" 50

# Check if we're in the right directory
if [ ! -d "../deploy/app" ]; then
    echo "❌ Error: Please run this script from the ai_net/test directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected structure: ai_net/test/ and ai_net/deploy/app/"
    exit 1
fi

# Conda environment detection and activation
CONDA_ENV_NAME="classifier"  # Default environment name
PYTHON_CMD=""

# Function to detect and activate conda
setup_python_environment() {
    # Check if micromamba is available (WSL environment)
    if command -v micromamba &> /dev/null; then
        echo "🐍 Micromamba detected (WSL environment)"
        
        # Check if we're already in a micromamba environment
        if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
            echo "   ✅ Already in micromamba environment: $CONDA_DEFAULT_ENV"
            PYTHON_CMD="python"
        else
            # Try to activate the specified environment
            echo "   🔄 Activating micromamba environment: $CONDA_ENV_NAME"
            
            # Initialize micromamba for bash
            eval "$(micromamba shell hook --shell bash)"
            
            # Activate environment
            if micromamba activate $CONDA_ENV_NAME 2>/dev/null; then
                echo "   ✅ Activated micromamba environment: $CONDA_ENV_NAME"
                PYTHON_CMD="python"
            else
                echo "   ⚠️  Failed to activate $CONDA_ENV_NAME, trying available environments"
                # Try to find available environments
                available_envs=$(micromamba env list | grep -v "^#" | awk '{print $1}' | grep -v "base")
                if [ ! -z "$available_envs" ]; then
                    first_env=$(echo "$available_envs" | head -1)
                    echo "   🔄 Trying environment: $first_env"
                    if micromamba activate $first_env 2>/dev/null; then
                        echo "   ✅ Activated micromamba environment: $first_env"
                        PYTHON_CMD="python"
                    else
                        echo "   ⚠️  Using base micromamba environment"
                        PYTHON_CMD="python"
                    fi
                else
                    echo "   ⚠️  Using base micromamba environment"
                    PYTHON_CMD="python"
                fi
            fi
        fi
    # Check if conda is available (Production environment)
    elif command -v conda &> /dev/null; then
        echo "🐍 Conda detected (Production environment)"
        
        # Check if we're already in a conda environment
        if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
            echo "   ✅ Already in conda environment: $CONDA_DEFAULT_ENV"
            PYTHON_CMD="python"
        else
            # Try to activate the specified environment
            echo "   🔄 Activating conda environment: $CONDA_ENV_NAME"
            
            # Initialize conda for bash
            eval "$(conda shell.bash hook)"
            
            # Activate environment
            if conda activate $CONDA_ENV_NAME 2>/dev/null; then
                echo "   ✅ Activated conda environment: $CONDA_ENV_NAME"
                PYTHON_CMD="python"
            else
                echo "   ⚠️  Failed to activate $CONDA_ENV_NAME, trying available environments"
                # Try to find available environments
                available_envs=$(conda env list | grep -v "^#" | awk '{print $1}' | grep -v "base")
                if [ ! -z "$available_envs" ]; then
                    first_env=$(echo "$available_envs" | head -1)
                    echo "   🔄 Trying environment: $first_env"
                    if conda activate $first_env 2>/dev/null; then
                        echo "   ✅ Activated conda environment: $first_env"
                        PYTHON_CMD="python"
                    else
                        echo "   ⚠️  Using base conda environment"
                        PYTHON_CMD="python"
                    fi
                else
                    echo "   ⚠️  Using base conda environment"
                    PYTHON_CMD="python"
                fi
            fi
        fi
    else
        echo "🐍 Neither conda nor micromamba found, using system Python"
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            echo "❌ Error: No Python found"
            exit 1
        fi
    fi
    
    echo "🐍 Using Python: $($PYTHON_CMD --version)"
}

# Setup Python environment
setup_python_environment

# Add deploy app to Python path for testing
export PYTHONPATH="../deploy:$PYTHONPATH"

echo "🔧 Available test scripts:"
echo "   1. backend_test.py - Interactive backend testing"
echo "   2. test_full_pipeline.py - Complete pipeline test"
echo "   3. test_backend_interactive.py - Step-by-step testing"
echo "   4. test_api.py - API endpoint testing"

echo ""
echo "📊 Quick Tests:"

# Test 1: Import validation
echo "🔍 Testing imports..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '../deploy')
sys.path.insert(0, '../deploy/app')
try:
    from app.pre_processor.excel_processor import ExcelProcessor
    from app.embedding.vectorization import Vectorization
    from app.util.vect_db_conn import VectorDBConnection
    from app.rag.rag_search import RAGSearcher, RAGBuilder
    print('   ✅ All imports successful')
except Exception as e:
    print(f'   ❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi

# Test 2: Database connection
echo "🗄️  Testing database..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '../deploy')
sys.path.insert(0, '../deploy/app')
try:
    from app.util.vect_db_conn import VectorDBConnection
    db = VectorDBConnection()
    db.close()
    print('   ✅ Database connection successful')
except Exception as e:
    print(f'   ⚠️  Database not available: {e}')
"

# Test 3: Embedding model
echo "🤖 Testing embedding model..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '../deploy')
sys.path.insert(0, '../deploy/app')
try:
    from app.embedding.vectorization import Vectorization
    vectorizer = Vectorization()
    test_embedding = vectorizer.encode_batch(['테스트 텍스트'], show_progress=False)
    print(f'   ✅ Embedding model working (shape: {test_embedding.shape})')
except Exception as e:
    print(f'   ❌ Embedding model error: {e}')
"

echo ""
echo "🎯 Test Environment Ready!"
echo "   Run individual test scripts as needed:"
echo "   - $PYTHON_CMD backend_test.py"
echo "   - $PYTHON_CMD test_full_pipeline.py"
echo "   - $PYTHON_CMD test_backend_interactive.py"
echo "   - $PYTHON_CMD test_api.py (requires running API server)"
