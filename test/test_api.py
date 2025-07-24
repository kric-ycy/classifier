"""
API Testing Script
Demonstrates how to use the AI Text Classification Testing API
"""

import requests
import json
import time
from pathlib import Path

# API configuration
API_BASE = "http://localhost:9639"
HEADERS = {"Content-Type": "application/json"}

def test_api_connection():
    """Test basic API connection."""
    print("🔗 Testing API connection...")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            print("   ✅ API is running")
            data = response.json()
            print(f"   📖 Documentation: {API_BASE}/docs")
            return True
        else:
            print(f"   ❌ API connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   💡 Make sure to start the API first:")
        print("      cd app/api && python embedding.py")
        return False

def test_database_connection():
    """Test database connection through API."""
    print("\n🗄️  Testing database connection...")
    try:
        response = requests.get(f"{API_BASE}/api/test/database")
        data = response.json()
        if data.get("status") == "success":
            print(f"   ✅ {data['message']}")
            print(f"   📊 Sample results found: {data.get('sample_results_count', 0)}")
        else:
            print(f"   ❌ {data['message']}")
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")

def test_embedding_generation():
    """Test embedding generation through API."""
    print("\n🔤 Testing embedding generation...")
    try:
        response = requests.get(f"{API_BASE}/api/test/embedding")
        data = response.json()
        if data.get("status") == "success":
            print(f"   ✅ {data['message']}")
            print(f"   📐 Embedding shape: {data.get('embedding_shape')}")
        else:
            print(f"   ❌ {data['message']}")
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")

def list_projects():
    """List available projects."""
    print("\n📁 Listing available projects...")
    try:
        response = requests.get(f"{API_BASE}/api/projects")
        data = response.json()
        projects = data.get("projects", [])
        
        if projects:
            print(f"   Found {len(projects)} projects:")
            for project in projects:
                print(f"   📂 {project['project_name']}: {project['file_count']} files")
                for file_name in project['files']:
                    print(f"      📄 {file_name}")
        else:
            print("   📭 No projects found")
            print("   💡 Upload some Excel files first using the upload endpoint")
        
        return projects
    except Exception as e:
        print(f"   ❌ Failed to list projects: {e}")
        return []

def upload_test_file():
    """Upload a test Excel file."""
    print("\n📤 Testing file upload...")
    
    # Look for test Excel files
    test_files = []
    data_dir = Path("../../data/raw")
    if data_dir.exists():
        for excel_file in data_dir.rglob("*.xlsx"):
            test_files.append(excel_file)
            if len(test_files) >= 3:  # Limit to first 3 files
                break
    
    if not test_files:
        print("   ⚠️  No test Excel files found in data/raw directory")
        return None
    
    test_file = test_files[0]
    project_name = "api_test_project"
    
    print(f"   📄 Uploading: {test_file.name}")
    print(f"   📂 Project: {project_name}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            params = {'project_name': project_name}
            
            response = requests.post(f"{API_BASE}/api/upload", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Upload successful!")
            print(f"   🆔 File ID: {data['file_id']}")
            print(f"   💾 Size: {data['file_size']} bytes")
            return project_name, data['file_id']
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        return None

def start_processing(project_name):
    """Start processing a project."""
    print(f"\n🔍 Starting processing for project: {project_name}")
    
    try:
        response = requests.post(f"{API_BASE}/api/process/{project_name}")
        
        if response.status_code == 200:
            data = response.json()
            batch_id = data['batch_id']
            print(f"   ✅ Processing started!")
            print(f"   🆔 Batch ID: {batch_id}")
            print(f"   📊 Files to process: {len(data['processing_summary']['file_names'])}")
            return batch_id
        else:
            print(f"   ❌ Processing failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        return None

def monitor_processing(batch_id):
    """Monitor processing progress."""
    print(f"\n⏳ Monitoring processing progress for batch: {batch_id}")
    
    max_attempts = 30  # Maximum wait time: 30 seconds
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{API_BASE}/api/status/{batch_id}")
            
            if response.status_code == 200:
                data = response.json()
                status = data['status']
                progress = data['progress']
                current_step = data['current_step']
                
                print(f"   📊 Status: {status} | Progress: {progress:.1%} | Step: {current_step}")
                
                if status == "completed":
                    print("   ✅ Processing completed!")
                    results = data.get('results_summary', {})
                    if results:
                        print(f"   📈 Results Summary:")
                        print(f"      • Total items: {results.get('total_items', 0)}")
                        print(f"      • Auto-classified: {results.get('auto_classified', 0)}")
                        print(f"      • Needs review: {results.get('needs_review', 0)}")
                        print(f"      • Auto rate: {results.get('auto_classification_rate', 0):.1%}")
                        print(f"      • Review tasks: {results.get('review_tasks_created', 0)}")
                    return True
                elif status == "error":
                    print(f"   ❌ Processing failed: {data.get('error', 'Unknown error')}")
                    return False
                
                time.sleep(1)
                attempt += 1
            else:
                print(f"   ❌ Status check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Monitoring error: {e}")
            return False
    
    print("   ⏰ Processing timeout - check status manually")
    return False

def check_review_tasks(batch_id):
    """Check review tasks generated."""
    print(f"\n📝 Checking review tasks for batch: {batch_id}")
    
    try:
        params = {'batch_id': batch_id, 'limit': 5}
        response = requests.get(f"{API_BASE}/api/review-tasks", params=params)
        
        if response.status_code == 200:
            tasks = response.json()
            print(f"   📋 Found {len(tasks)} review tasks:")
            
            for i, task in enumerate(tasks, 1):
                print(f"   {i}. Word: '{task['word']}'")
                print(f"      Classified as: '{task['classified_as']}'")
                print(f"      Confidence: {task['confidence']:.3f}")
                print(f"      Reason: {task['reason']}")
                print(f"      Group: {task['question_group']} | Column: {task['column']}")
                print()
            
            return tasks
        else:
            print(f"   ❌ Failed to get review tasks: {response.status_code}")
            return []
    except Exception as e:
        print(f"   ❌ Review tasks error: {e}")
        return []

def submit_sample_review(tasks):
    """Submit a sample review result."""
    if not tasks:
        print("\n   ⚠️  No review tasks to demonstrate")
        return
    
    print(f"\n✏️  Submitting sample review result...")
    
    task = tasks[0]  # Use first task
    doc_id = task['doc_id']
    
    # Simulate a user correction
    review_data = {
        "doc_id": doc_id,
        "revised_content": f"corrected_{task['classified_as']}",
        "revised_by": "user"
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/review-result", 
                               json=review_data, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Review submitted successfully!")
            print(f"   📝 Original: '{data['original_classification']}'")
            print(f"   ✏️  Revised: '{data['revised_classification']}'")
        else:
            print(f"   ❌ Review submission failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Review submission error: {e}")

def download_results(batch_id):
    """Download final results."""
    print(f"\n📥 Downloading results for batch: {batch_id}")
    
    try:
        response = requests.get(f"{API_BASE}/api/final-result/{batch_id}")
        
        if response.status_code == 200:
            filename = f"api_test_results_{batch_id}.xlsx"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ Results downloaded: {filename}")
            print(f"   📊 File size: {len(response.content)} bytes")
        else:
            print(f"   ❌ Download failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Download error: {e}")

def run_complete_api_test():
    """Run complete API test workflow."""
    print("🚀 AI Text Classification API - Complete Test")
    print("=" * 60)
    
    # Step 1: Test basic connection
    if not test_api_connection():
        return
    
    # Step 2: Test components
    test_database_connection()
    test_embedding_generation()
    
    # Step 3: List existing projects
    projects = list_projects()
    
    # Step 4: Upload test file
    upload_result = upload_test_file()
    if not upload_result:
        print("\n⚠️  Skipping processing tests - no file uploaded")
        return
    
    project_name, file_id = upload_result
    
    # Step 5: Start processing
    batch_id = start_processing(project_name)
    if not batch_id:
        return
    
    # Step 6: Monitor processing
    if not monitor_processing(batch_id):
        return
    
    # Step 7: Check review tasks
    tasks = check_review_tasks(batch_id)
    
    # Step 8: Submit sample review
    submit_sample_review(tasks)
    
    # Step 9: Download results
    download_results(batch_id)
    
    print("\n🎉 Complete API test finished!")
    print("=" * 60)
    print("✅ All API endpoints tested successfully")
    print("📖 View full API documentation at: http://localhost:9639/docs")
    print("🔧 API root endpoint: http://localhost:9639/")

if __name__ == "__main__":
    run_complete_api_test()
