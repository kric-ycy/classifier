from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import uvicorn
import os
import sys
import pandas as pd
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment configuration
load_dotenv()

# Add the project root and app directory to Python path
project_root = Path(__file__).parent.parent.parent
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

# Import our backend components
from app.pre_processor.excel_processor import ExcelProcessor
from app.embedding.vectorization import Vectorization
from app.util.vect_db_conn import VectorDBConnection
from app.rag.rag_search import RAGSearcher, RAGBuilder

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Add file logging in production
log_file = os.getenv("LOG_FILE")
if log_file:
    logger.add(log_file, rotation="10 MB", retention="30 days", level="INFO")


app = FastAPI(title="AI Text Classification Testing API")

# Initialize backend components
ep = ExcelProcessor()
vectorizer = Vectorization()
rag_searcher = RAGSearcher(similarity_threshold=0.24)

# Global storage for processing results (in production, use a database)
processing_results = {}
review_tasks = {}

# Create temp_data directory if it doesn't exist
temp_data_dir = Path("../temp_data")
temp_data_dir.mkdir(exist_ok=True)

# -----------------------------
# 📁 Schemas
# -----------------------------

class UploadResponse(BaseModel):
    file_id: str
    message: str
    project_name: str
    file_size: int
    timestamp: str

class ProcessingResponse(BaseModel):
    batch_id: str
    status: str
    message: str
    processing_summary: Dict[str, Any]

class ReviewTask(BaseModel):
    doc_id: str
    word: str
    classified_as: str
    confidence: float
    reason: str
    question_group: str
    column: str

class ReviewRequest(BaseModel):
    doc_id: str
    revised_content: str
    revised_by: str  # "user" or "ai"

class DomainRequest(BaseModel):
    doc_id: str
    domain_description: str

class ProcessingStatus(BaseModel):
    batch_id: str
    status: str
    progress: float
    current_step: str
    results_summary: Optional[Dict[str, Any]] = None

# -----------------------------
# 📤 Upload
# -----------------------------

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(project_name: str, file: UploadFile = File(...)):
    """Upload Excel file for processing."""
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")
        
        # Create project directory
        project_dir = temp_data_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Generate unique file ID with timestamp
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save file with original extension
        file_extension = Path(file.filename).suffix
        stored_filename = f"{timestamp}_{file_id}{file_extension}"
        file_path = project_dir / stored_filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        
        return UploadResponse(
            file_id=file_id,
            message=f"File '{file.filename}' uploaded successfully",
            project_name=project_name,
            file_size=file_size,
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/api/projects")
async def list_projects():
    """List all available projects."""
    try:
        projects = []
        if temp_data_dir.exists():
            for project_dir in temp_data_dir.iterdir():
                if project_dir.is_dir():
                    files = [f.name for f in project_dir.iterdir() if f.suffix in ['.xlsx', '.xls']]
                    projects.append({
                        "project_name": project_dir.name,
                        "file_count": len(files),
                        "files": files
                    })
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

# -----------------------------
# 🔍 RAG Processing
# -----------------------------

async def process_file_background(project_name: str, batch_id: str):
    """Background processing function."""
    try:
        # Update status
        processing_results[batch_id] = {
            "status": "processing",
            "progress": 0.1,
            "current_step": "Reading Excel files",
            "project_name": project_name
        }
        
        # Find Excel files
        project_dir = temp_data_dir / project_name
        excel_files = [f for f in project_dir.iterdir() if f.suffix in ['.xlsx', '.xls']]
        
        if not excel_files:
            processing_results[batch_id]["status"] = "error"
            processing_results[batch_id]["error"] = "No Excel files found"
            return
        
        all_results = []
        
        for i, excel_file in enumerate(excel_files):
            # Update progress
            progress = 0.1 + (i / len(excel_files)) * 0.8
            processing_results[batch_id]["progress"] = progress
            processing_results[batch_id]["current_step"] = f"Processing {excel_file.name}"
            
            # Parse Excel file
            raw_df, obj_data, grouped_dict = ep.read_excel(str(excel_file))
            
            # Process first 3 question groups (for testing)
            for q_num in list(grouped_dict.keys())[:3]:
                columns = grouped_dict[q_num]
                if not columns:
                    continue
                
                # Get question data
                q_data = ep.get_question_columns(obj_data, q_num, grouped_dict)
                if q_data.empty:
                    continue
                
                # Process first 2 columns per question group
                for col in columns[:2]:
                    if col in q_data.columns:
                        # Get unique values (limit for testing)
                        unique_values = q_data[col].dropna().astype(str).unique()[:20]
                        
                        if len(unique_values) > 0:
                            # Perform RAG search
                            col_results = rag_searcher.search_batch(unique_values.tolist(), k=3)
                            
                            # Add metadata
                            for result in col_results:
                                result['question_group'] = q_num
                                result['column'] = col
                                result['file'] = excel_file.name
                                result['batch_id'] = batch_id
                            
                            all_results.extend(col_results)
        
        # Analyze results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Apply threshold filtering
            auto_classified, needs_review = rag_searcher.filter_by_threshold(results_df)
            
            # Get summary statistics
            summary = rag_searcher.create_classification_summary(results_df)
            
            # Get review candidates
            review_candidates = rag_searcher.get_review_candidates(results_df, top_n=10)
            
            # Store review tasks
            for _, row in review_candidates.iterrows():
                doc_id = str(uuid.uuid4())
                review_tasks[doc_id] = {
                    "doc_id": doc_id,
                    "word": row['word'],
                    "classified_as": row['classified'],
                    "confidence": 1 - row['distance'],  # Convert distance to confidence
                    "reason": f"Low similarity score ({row['distance']:.3f})",
                    "question_group": row['question_group'],
                    "column": row['column'],
                    "batch_id": batch_id,
                    "created_at": datetime.now().isoformat()
                }
            
            # Final results
            processing_results[batch_id] = {
                "status": "completed",
                "progress": 1.0,
                "current_step": "Processing completed",
                "project_name": project_name,
                "results_summary": {
                    "total_items": len(all_results),
                    "auto_classified": len(auto_classified),
                    "needs_review": len(needs_review),
                    "auto_classification_rate": summary["auto_classification_rate"],
                    "avg_confidence": summary["avg_confidence"],
                    "files_processed": len(excel_files),
                    "review_tasks_created": len(review_candidates)
                }
            }
        else:
            processing_results[batch_id] = {
                "status": "completed",
                "progress": 1.0,
                "current_step": "No items found for processing",
                "project_name": project_name,
                "results_summary": {
                    "total_items": 0,
                    "message": "No processable data found"
                }
            }
            
    except Exception as e:
        processing_results[batch_id] = {
            "status": "error",
            "progress": 0.0,
            "current_step": "Processing failed",
            "error": str(e),
            "project_name": project_name
        }

@app.post("/api/process/{project_name}", response_model=ProcessingResponse)
async def process_file(project_name: str, background_tasks: BackgroundTasks):
    """Start processing files in a project."""
    try:
        # Check if project exists
        project_dir = temp_data_dir / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
        
        # Check for Excel files
        excel_files = [f for f in project_dir.iterdir() if f.suffix in ['.xlsx', '.xls']]
        if not excel_files:
            raise HTTPException(status_code=400, detail="No Excel files found for processing")
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Start background processing
        background_tasks.add_task(process_file_background, project_name, batch_id)
        
        return ProcessingResponse(
            batch_id=batch_id,
            status="started",
            message=f"Processing started for {len(excel_files)} files",
            processing_summary={
                "files_found": len(excel_files),
                "file_names": [f.name for f in excel_files]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.get("/api/status/{batch_id}", response_model=ProcessingStatus)
async def get_processing_status(batch_id: str):
    """Get processing status for a batch."""
    if batch_id not in processing_results:
        raise HTTPException(status_code=404, detail="Batch ID not found")
    
    result = processing_results[batch_id]
    return ProcessingStatus(
        batch_id=batch_id,
        status=result["status"],
        progress=result["progress"],
        current_step=result["current_step"],
        results_summary=result.get("results_summary")
    )

# -----------------------------
# 📝 Review 관련
# -----------------------------

@app.get("/api/review-tasks", response_model=List[ReviewTask])
async def get_review_tasks(batch_id: Optional[str] = None, limit: int = 20):
    """Get items that need human review."""
    try:
        tasks = []
        for doc_id, task_data in review_tasks.items():
            # Filter by batch_id if provided
            if batch_id and task_data.get("batch_id") != batch_id:
                continue
            
            tasks.append(ReviewTask(
                doc_id=task_data["doc_id"],
                word=task_data["word"],
                classified_as=task_data["classified_as"],
                confidence=task_data["confidence"],
                reason=task_data["reason"],
                question_group=task_data["question_group"],
                column=task_data["column"]
            ))
            
            if len(tasks) >= limit:
                break
        
        return tasks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get review tasks: {str(e)}")

@app.post("/api/review-result")
async def submit_review_result(request: ReviewRequest):
    """Submit reviewed classification result."""
    try:
        if request.doc_id not in review_tasks:
            raise HTTPException(status_code=404, detail="Review task not found")
        
        # Update the review task
        task = review_tasks[request.doc_id]
        task["reviewed"] = True
        task["revised_content"] = request.revised_content
        task["revised_by"] = request.revised_by
        task["reviewed_at"] = datetime.now().isoformat()
        
        # Here you would typically:
        # 1. Store the corrected classification in the RAG database
        # 2. Update the knowledge base
        # 3. Retrain or update the model
        
        return {
            "status": "received",
            "doc_id": request.doc_id,
            "message": "Review result processed successfully",
            "original_word": task["word"],
            "original_classification": task["classified_as"],
            "revised_classification": request.revised_content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process review result: {str(e)}")

@app.get("/api/review-stats/{batch_id}")
async def get_review_stats(batch_id: str):
    """Get review statistics for a batch."""
    try:
        total_tasks = sum(1 for task in review_tasks.values() if task.get("batch_id") == batch_id)
        reviewed_tasks = sum(1 for task in review_tasks.values() 
                           if task.get("batch_id") == batch_id and task.get("reviewed", False))
        
        return {
            "batch_id": batch_id,
            "total_review_tasks": total_tasks,
            "completed_reviews": reviewed_tasks,
            "pending_reviews": total_tasks - reviewed_tasks,
            "completion_rate": reviewed_tasks / total_tasks if total_tasks > 0 else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get review stats: {str(e)}")

# -----------------------------
# 🌐 도메인 요청
# -----------------------------

@app.post("/api/domain-request")
async def request_domain_extension(request: DomainRequest):
    """Request addition of new domain/classification category."""
    try:
        # Store domain request (in production, save to database)
        domain_id = str(uuid.uuid4())
        domain_request = {
            "domain_id": domain_id,
            "doc_id": request.doc_id,
            "domain_description": request.domain_description,
            "requested_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Here you would typically:
        # 1. Add the new domain to your classification schema
        # 2. Update the RAG database with new categories
        # 3. Notify administrators for review
        
        return {
            "status": "domain_requested",
            "domain_id": domain_id,
            "doc_id": request.doc_id,
            "message": "Domain extension request submitted for review"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process domain request: {str(e)}")

# -----------------------------
# 📦 결과 다운로드
# -----------------------------

@app.get("/api/final-result/{batch_id}")
async def download_final_result(batch_id: str):
    """Download final classification results as Excel file."""
    try:
        if batch_id not in processing_results:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        batch_data = processing_results[batch_id]
        if batch_data["status"] != "completed":
            raise HTTPException(status_code=400, detail="Processing not completed yet")
        
        # Create results directory
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate Excel file with results
        file_path = results_dir / f"{batch_id}_final.xlsx"
        
        # Create summary data
        summary_data = {
            "Batch ID": [batch_id],
            "Project": [batch_data["project_name"]],
            "Status": [batch_data["status"]],
            "Total Items": [batch_data.get("results_summary", {}).get("total_items", 0)],
            "Auto Classified": [batch_data.get("results_summary", {}).get("auto_classified", 0)],
            "Needs Review": [batch_data.get("results_summary", {}).get("needs_review", 0)],
            "Auto Rate": [batch_data.get("results_summary", {}).get("auto_classification_rate", 0)],
            "Avg Confidence": [batch_data.get("results_summary", {}).get("avg_confidence", 0)]
        }
        
        # Create review tasks data
        review_data = []
        for task in review_tasks.values():
            if task.get("batch_id") == batch_id:
                review_data.append({
                    "Doc ID": task["doc_id"],
                    "Word": task["word"],
                    "Classified As": task["classified_as"],
                    "Confidence": task["confidence"],
                    "Question Group": task["question_group"],
                    "Column": task["column"],
                    "Reviewed": task.get("reviewed", False),
                    "Revised Content": task.get("revised_content", ""),
                    "Revised By": task.get("revised_by", "")
                })
        
        # Write to Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            if review_data:
                pd.DataFrame(review_data).to_excel(writer, sheet_name='Review Tasks', index=False)
        
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"classification_results_{batch_id}.xlsx"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate results file: {str(e)}")

# -----------------------------
# 🔧 Testing & Debug Endpoints
# -----------------------------

@app.get("/api/test/database")
async def test_database_connection():
    """Test database connection."""
    try:
        db = VectorDBConnection()
        
        # Test simple query
        test_embedding = "[0.1]" + ", 0.0" * 767
        results = db.search_similar(test_embedding, k=1)
        
        db.close()
        
        return {
            "status": "success",
            "message": "Database connection successful",
            "sample_results_count": len(results)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        }

@app.get("/api/test/embedding")
async def test_embedding():
    """Test embedding generation."""
    try:
        test_text = "테스트 문장입니다"
        embedding = vectorizer.encode(test_text, normalize=True, return_numpy=True)
        
        return {
            "status": "success",
            "message": "Embedding generation successful",
            "embedding_shape": embedding.shape,
            "sample_text": test_text
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Embedding generation failed: {str(e)}"
        }

@app.get("/api/debug/processing-results")
async def get_all_processing_results():
    """Get all processing results for debugging."""
    return {
        "processing_results": processing_results,
        "review_tasks_count": len(review_tasks)
    }

@app.delete("/api/debug/clear-data")
async def clear_debug_data():
    """Clear all processing data (for testing)."""
    global processing_results, review_tasks
    processing_results = {}
    review_tasks = {}
    return {"message": "All data cleared"}

@app.get("/")
async def root():
    """API root endpoint with usage information."""
    return {
        "message": "AI Text Classification Testing API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /api/upload - Upload Excel files",
            "process": "POST /api/process/{project_name} - Start processing",
            "status": "GET /api/status/{batch_id} - Check processing status",
            "review_tasks": "GET /api/review-tasks - Get review tasks",
            "review_result": "POST /api/review-result - Submit review results",
            "final_result": "GET /api/final-result/{batch_id} - Download results",
            "projects": "GET /api/projects - List all projects",
            "test_db": "GET /api/test/database - Test database connection",
            "test_embedding": "GET /api/test/embedding - Test embedding generation"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    print("🚀 Starting AI Text Classification Testing API...")
    print("📖 API Documentation: http://localhost:9639/docs")
    print("🔧 Root endpoint: http://localhost:9639/")
    uvicorn.run("embedding:app", host="0.0.0.0", port=9639, reload=True)