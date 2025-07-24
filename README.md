# AI Text Classification - Production Ready System

## 🏗️ Project Structure

```
ai_text_classification/
├── ai_net/                          # Production-ready system
│   ├── test/                        # Test environment
│   │   ├── start_test_env.sh        # Test environment setup
│   │   ├── backend_test.py          # Interactive backend testing
│   │   ├── test_full_pipeline.py    # Complete pipeline test
│   │   ├── test_backend_interactive.py # Step-by-step testing
│   │   └── test_api.py              # API endpoint testing
│   ├── deploy/                      # Production deployment
│   │   ├── app/                     # Application code
│   │   │   ├── api/                 # FastAPI endpoints
│   │   │   ├── pre_processor/       # Excel processing
│   │   │   ├── embedding/           # Text embeddings
│   │   │   ├── rag/                 # RAG search system
│   │   │   └── util/                # Database utilities
│   │   ├── start_production.sh      # Production server startup
│   │   ├── requirements.txt         # Production dependencies
│   │   └── .env.example             # Environment configuration
│   ├── detect_environment.sh        # Environment detection
│   └── check_structure.sh           # Project structure overview
├── data/                            # Data files
└── [legacy files]                   # Original development files
```

## 🚀 Quick Start

### 1. **Environment Detection**
```bash
# Check your environment and available tools
./ai_net/detect_environment.sh
```

### 2. **Testing Environment (Development)**
```bash
cd ai_net/test
./start_test_env.sh

# Run individual tests
python test_backend_interactive.py   # Interactive step-by-step testing
python backend_test.py              # Complete pipeline testing
python test_api.py                  # API testing (requires server)
```

### 3. **Production Deployment**
```bash
cd ai_net/deploy

# Configure environment (first time only)
cp .env.example .env
# Edit .env with your database credentials

# Start production server
./start_production.sh
```

## 🌍 Environment Support

### **WSL + Micromamba** (Development)
- ✅ Automatic micromamba environment activation
- ✅ Optimized for development and testing
- 🔧 Uses `classifier` environment by default

### **Linux/Production + Conda**
- ✅ Full conda environment support
- ✅ Production-ready deployment
- 🔧 Multi-worker FastAPI server

### **Fallback**
- ✅ System Python support if no conda/micromamba

## 📊 Features

### **Complete Backend Pipeline**
- ✅ Excel file parsing and processing
- ✅ Korean text embedding (SBERT)
- ✅ PostgreSQL vector database
- ✅ RAG-based classification
- ✅ Human-in-the-loop review system

### **Production API**
- ✅ FastAPI with async processing
- ✅ File upload and validation
- ✅ Background task processing
- ✅ Real-time status monitoring
- ✅ Result download endpoints

### **Testing Infrastructure**
- ✅ Component-level testing
- ✅ Integration testing
- ✅ Interactive debugging tools
- ✅ API endpoint validation

## 🔧 Configuration

### **Database Setup**
```bash
# PostgreSQL with pgvector extension required
# Configure in ai_net/deploy/.env:
DATABASE_URL=postgresql://user:pass@localhost:5432/ai_classification
```

### **Environment Variables**
```bash
# Copy and edit configuration
cd ai_net/deploy
cp .env.example .env
# Edit .env with your settings
```

## 🧪 Testing Workflow

1. **Environment Setup**: `./ai_net/detect_environment.sh`
2. **Quick Tests**: `cd ai_net/test && ./start_test_env.sh`
3. **Interactive Testing**: `python test_backend_interactive.py`
4. **Full Pipeline**: `python backend_test.py`
5. **API Testing**: `python test_api.py`

## 🚀 Production Workflow

1. **Environment Setup**: `cd ai_net/deploy`
2. **Configuration**: Edit `.env` file
3. **Dependencies**: Auto-installed by startup script
4. **Launch**: `./start_production.sh`
5. **Access**: `http://localhost:8000/docs`

## 📈 Monitoring

- **Logs**: `ai_net/deploy/logs/api.log`
- **Health Check**: `GET /debug/test_db`
- **Metrics**: Available through FastAPI endpoints

---

**Ready for production!** 🎉

The system supports both development (WSL + micromamba) and production (Linux + conda) environments with automatic detection and configuration.
