#!/bin/bash

# AI Text Classification - Production Deployment Script

echo "🚀 AI Text Classification - Production Deployment"
echo "=" 50

# Check if we're in the right directory
if [ ! -d "app" ]; then
    echo "❌ Error: Please run this script from the ai_net/deploy directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected structure: ai_net/deploy/app/api/embedding.py"
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

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "   ⚠️  Please configure your .env file with proper database credentials"
fi

# Create necessary directories
echo "📁 Creating production directories..."
mkdir -p logs
mkdir -p models
mkdir -p temp_data
mkdir -p results
mkdir -p uploads
echo "   ✅ Directories created"

# Install dependencies
echo "🔧 Installing production dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "   ✅ Dependencies installed"

# Check database connection
echo "🗄️  Testing database connection..."
$PYTHON_CMD -c "
import sys
sys.path.append('app')
try:
    from app.util.vect_db_conn import VectorDBConnection
    db = VectorDBConnection()
    db.close()
    print('   ✅ Database connection successful')
except Exception as e:
    print(f'   ❌ Database connection failed: {e}')
    print('   💡 Please check your database configuration in .env')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Load environment variables
source .env 2>/dev/null || true

# Set defaults if not in environment
API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-9639}
API_WORKERS=${API_WORKERS:-4}

echo "🌐 Starting production server..."
echo "   📖 API Documentation: http://$API_HOST:$API_PORT/docs"
echo "   🔧 API Root: http://$API_HOST:$API_PORT/"
echo "   👥 Workers: $API_WORKERS"
echo "   📊 Logs: logs/api.log"
echo "   🛑 Press Ctrl+C to stop the server"
echo ""

# Start the production server with uvicorn
cd app/api
python embedding.py 