"""
Backend Testing Script
Test the complete AI text classification pipeline with real data
"""

import pandas as pd
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add deploy app directory to path for testing
project_root = Path(__file__).parent.parent
deploy_dir = project_root / "deploy"
app_dir = deploy_dir / "app"

sys.path.insert(0, str(deploy_dir))
sys.path.insert(0, str(app_dir))

from app.pre_processor.excel_processor import ExcelProcessor
from app.embedding.vectorization import Vectorization
from app.util.vect_db_conn import VectorDBConnection
from app.rag.rag_search import RAGSearcher, RAGBuilder


class AITextClassificationBackend:
    """
    Complete backend logic for AI text classification system.
    Handles the full pipeline from Excel upload to classification results.
    """
    
    def __init__(self, similarity_threshold=0.24, temp_dir="temp_data"):
        """
        Initialize the backend system.
        
        Parameters:
        similarity_threshold (float): Threshold for auto-classification vs human review
        temp_dir (str): Directory for temporary file storage
        """
        self.similarity_threshold = similarity_threshold
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.excel_processor = ExcelProcessor()
        self.vectorizer = Vectorization()
        self.rag_searcher = RAGSearcher(similarity_threshold=similarity_threshold)
        self.rag_builder = RAGBuilder()
        
        print(f"🚀 AI Text Classification Backend initialized")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Temp directory: {self.temp_dir}")
    
    def store_file(self, file_path: str, project_name: str) -> Dict[str, Any]:
        """
        Store uploaded file in project directory.
        
        Parameters:
        file_path (str): Path to the source file
        project_name (str): Project identifier
        
        Returns:
        Dict: Storage result with metadata
        """
        try:
            # Create project directory
            project_dir = self.temp_dir / project_name
            project_dir.mkdir(exist_ok=True)
            
            # Copy file to project directory
            source_path = Path(file_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_filename = f"{timestamp}_{source_path.name}"
            dest_path = project_dir / dest_filename
            
            # Copy file content
            import shutil
            shutil.copy2(source_path, dest_path)
            
            file_info = {
                "status": "success",
                "project_name": project_name,
                "original_filename": source_path.name,
                "stored_filename": dest_filename,
                "stored_path": str(dest_path),
                "file_size": dest_path.stat().st_size,
                "timestamp": timestamp
            }
            
            print(f"✅ File stored successfully: {dest_filename}")
            return file_info
            
        except Exception as e:
            error_info = {
                "status": "error",
                "error": str(e),
                "project_name": project_name
            }
            print(f"❌ File storage failed: {e}")
            return error_info
    
    def parse_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse Excel file and extract text columns.
        
        Parameters:
        file_path (str): Path to Excel file
        
        Returns:
        Dict: Parsing results with extracted data
        """
        try:
            print(f"📊 Parsing Excel file: {Path(file_path).name}")
            
            # Read and process Excel file
            raw_df, obj_data, grouped_dict = self.excel_processor.read_excel(file_path)
            
            # Get statistics
            total_columns = len(raw_df.columns)
            text_columns = len(obj_data.columns)
            question_groups = len(grouped_dict)
            total_rows = len(obj_data)
            
            # Sample data for preview
            sample_data = {}
            for q_num, columns in list(grouped_dict.items())[:3]:  # First 3 groups
                if columns:
                    sample_df = self.excel_processor.get_question_columns(obj_data, q_num, grouped_dict)
                    sample_data[q_num] = {
                        "columns": columns,
                        "sample_values": sample_df.iloc[:5].fillna("").to_dict() if not sample_df.empty else {}
                    }
            
            parse_result = {
                "status": "success",
                "file_path": file_path,
                "total_columns": total_columns,
                "text_columns": text_columns,
                "question_groups": question_groups,
                "total_rows": total_rows,
                "grouped_dict": grouped_dict,
                "sample_data": sample_data,
                "obj_data_shape": obj_data.shape
            }
            
            print(f"   ✅ Parsed {text_columns} text columns from {total_columns} total columns")
            print(f"   ✅ Found {question_groups} question groups with {total_rows} data rows")
            
            return parse_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "file_path": file_path
            }
            print(f"❌ Excel parsing failed: {e}")
            return error_result
    
    def compare_with_rag(self, obj_data: pd.DataFrame, grouped_dict: Dict, 
                        question_numbers: List[str] = None) -> Dict[str, Any]:
        """
        Compare extracted text with RAG database using semantic similarity.
        
        Parameters:
        obj_data (pd.DataFrame): Extracted text data
        grouped_dict (Dict): Question grouping information
        question_numbers (List[str]): Specific questions to process (None for all)
        
        Returns:
        Dict: RAG comparison results
        """
        try:
            print(f"🔍 Comparing with RAG database...")
            
            # Process specific questions or all questions
            questions_to_process = question_numbers or list(grouped_dict.keys())
            all_results = []
            processing_summary = {}
            
            for q_num in questions_to_process[:3]:  # Limit to first 3 for testing
                print(f"   Processing question group: {q_num}")
                
                # Get question data
                q_data = self.excel_processor.get_question_columns(obj_data, q_num, grouped_dict)
                if q_data.empty:
                    continue
                
                # Get columns for this question
                q_columns = grouped_dict.get(q_num, [])
                
                # Process each column
                for col in q_columns[:2]:  # Limit to first 2 columns per question for testing
                    if col in q_data.columns:
                        print(f"     Processing column: {col}")
                        
                        # Get non-null values
                        col_data = q_data[col].dropna().astype(str)
                        unique_values = col_data.unique()
                        
                        if len(unique_values) > 0:
                            # Limit to first 10 unique values for testing
                            test_values = unique_values[:10]
                            
                            # Search in RAG
                            col_results = self.rag_searcher.search_batch(test_values.tolist(), k=3)
                            
                            # Add metadata
                            for result in col_results:
                                result['question_group'] = q_num
                                result['column'] = col
                            
                            all_results.extend(col_results)
                            
                            processing_summary[f"{q_num}_{col}"] = {
                                "total_values": len(col_data),
                                "unique_values": len(unique_values),
                                "processed_values": len(test_values),
                                "rag_results": len(col_results)
                            }
            
            # Convert to DataFrame for analysis
            if all_results:
                results_df = pd.DataFrame(all_results)
                
                # Apply threshold filtering
                auto_classified, needs_review = self.rag_searcher.filter_by_threshold(results_df)
                
                # Get summary statistics
                summary = self.rag_searcher.create_classification_summary(results_df)
                
                # Get review candidates
                review_candidates = self.rag_searcher.get_review_candidates(results_df, top_n=5)
                
                rag_result = {
                    "status": "success",
                    "total_items_processed": len(all_results),
                    "auto_classified_count": len(auto_classified),
                    "needs_review_count": len(needs_review),
                    "auto_classification_rate": summary["auto_classification_rate"],
                    "avg_confidence": summary["avg_confidence"],
                    "processing_summary": processing_summary,
                    "review_candidates": review_candidates.to_dict('records') if not review_candidates.empty else [],
                    "sample_auto_classified": auto_classified.head(5).to_dict('records') if not auto_classified.empty else [],
                    "sample_needs_review": needs_review.head(5).to_dict('records') if not needs_review.empty else []
                }
                
                print(f"   ✅ Processed {len(all_results)} items")
                print(f"   ✅ Auto-classified: {len(auto_classified)} ({summary['auto_classification_rate']:.1%})")
                print(f"   ✅ Needs review: {len(needs_review)}")
                
            else:
                rag_result = {
                    "status": "no_results",
                    "message": "No items found for RAG comparison",
                    "processing_summary": processing_summary
                }
                print("   ⚠️  No items found for RAG comparison")
            
            return rag_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ RAG comparison failed: {e}")
            return error_result
    
    def create_results_summary(self, project_name: str, file_info: Dict, 
                              parse_result: Dict, rag_result: Dict) -> Dict[str, Any]:
        """
        Create comprehensive results summary.
        
        Parameters:
        project_name (str): Project identifier
        file_info (Dict): File storage information
        parse_result (Dict): Excel parsing results
        rag_result (Dict): RAG comparison results
        
        Returns:
        Dict: Complete results summary
        """
        summary = {
            "project_name": project_name,
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": "completed",
            "file_info": file_info,
            "parsing": {
                "status": parse_result.get("status"),
                "total_columns": parse_result.get("total_columns", 0),
                "text_columns": parse_result.get("text_columns", 0),
                "question_groups": parse_result.get("question_groups", 0),
                "total_rows": parse_result.get("total_rows", 0)
            },
            "rag_analysis": {
                "status": rag_result.get("status"),
                "total_processed": rag_result.get("total_items_processed", 0),
                "auto_classified": rag_result.get("auto_classified_count", 0),
                "needs_review": rag_result.get("needs_review_count", 0),
                "auto_rate": rag_result.get("auto_classification_rate", 0),
                "avg_confidence": rag_result.get("avg_confidence", 0)
            },
            "review_items": rag_result.get("review_candidates", []),
            "next_steps": []
        }
        
        # Add next steps based on results
        if rag_result.get("needs_review_count", 0) > 0:
            summary["next_steps"].append(f"Review {rag_result['needs_review_count']} items requiring human validation")
        
        if rag_result.get("auto_classified_count", 0) > 0:
            summary["next_steps"].append(f"Export {rag_result['auto_classified_count']} auto-classified items")
        
        if not summary["next_steps"]:
            summary["next_steps"].append("No items found for processing")
        
        return summary
    
    def process_complete_pipeline(self, file_path: str, project_name: str) -> Dict[str, Any]:
        """
        Execute the complete processing pipeline.
        
        Parameters:
        file_path (str): Path to Excel file
        project_name (str): Project identifier
        
        Returns:
        Dict: Complete pipeline results
        """
        print(f"\n🎯 Starting complete pipeline for project: {project_name}")
        print("=" * 60)
        
        # Step 1: Store file
        print("📁 Step 1: Storing file...")
        file_info = self.store_file(file_path, project_name)
        if file_info["status"] != "success":
            return {"status": "failed", "step": "file_storage", "error": file_info}
        
        # Step 2: Parse Excel
        print("\n📊 Step 2: Parsing Excel file...")
        stored_path = file_info["stored_path"]
        parse_result = self.parse_excel_file(stored_path)
        if parse_result["status"] != "success":
            return {"status": "failed", "step": "excel_parsing", "error": parse_result}
        
        # Step 3: Compare with RAG
        print("\n🔍 Step 3: Comparing with RAG database...")
        # Reload the data for RAG comparison
        raw_df, obj_data, grouped_dict = self.excel_processor.read_excel(stored_path)
        rag_result = self.compare_with_rag(obj_data, grouped_dict)
        
        # Step 4: Create summary
        print("\n📋 Step 4: Creating results summary...")
        summary = self.create_results_summary(project_name, file_info, parse_result, rag_result)
        
        print("\n✅ Pipeline completed successfully!")
        print("=" * 60)
        
        return {
            "status": "success",
            "summary": summary,
            "detailed_results": {
                "file_info": file_info,
                "parse_result": parse_result,
                "rag_result": rag_result
            }
        }
    
    def save_results(self, results: Dict, output_path: str = None) -> str:
        """
        Save results to JSON file.
        
        Parameters:
        results (Dict): Results to save
        output_path (str): Output file path (None for auto-generation)
        
        Returns:
        str: Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = results.get("summary", {}).get("project_name", "unknown")
            output_path = f"results_{project_name}_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return ""


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="AI Text Classification Backend Testing")
    parser.add_argument("file_path", help="Path to Excel file to process")
    parser.add_argument("project_name", help="Project name identifier")
    parser.add_argument("--threshold", type=float, default=0.24, 
                       help="Similarity threshold for classification (default: 0.24)")
    parser.add_argument("--temp-dir", default="temp_data", 
                       help="Temporary directory for file storage")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize backend
    backend = AITextClassificationBackend(
        similarity_threshold=args.threshold,
        temp_dir=args.temp_dir
    )
    
    try:
        # Process complete pipeline
        results = backend.process_complete_pipeline(args.file_path, args.project_name)
        
        # Print summary
        if results["status"] == "success":
            summary = results["summary"]
            print(f"\n📊 PIPELINE SUMMARY")
            print(f"   Project: {summary['project_name']}")
            print(f"   File: {summary['file_info']['original_filename']}")
            print(f"   Text columns: {summary['parsing']['text_columns']}")
            print(f"   Question groups: {summary['parsing']['question_groups']}")
            print(f"   Items processed: {summary['rag_analysis']['total_processed']}")
            print(f"   Auto-classified: {summary['rag_analysis']['auto_classified']} ({summary['rag_analysis']['auto_rate']:.1%})")
            print(f"   Needs review: {summary['rag_analysis']['needs_review']}")
            print(f"   Next steps: {len(summary['next_steps'])} actions")
            
            for i, step in enumerate(summary['next_steps'], 1):
                print(f"     {i}. {step}")
                
        else:
            print(f"\n❌ Pipeline failed at step: {results.get('step', 'unknown')}")
            print(f"   Error: {results.get('error', 'Unknown error')}")
        
        # Save results if requested
        if args.save_results:
            backend.save_results(results, args.output)
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
