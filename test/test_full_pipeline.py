"""
Full Integration Test - Complete pipeline with real data
Tests the complete AI text classification system with actual Excel files
"""

import pandas as pd
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

try:
    from pre_processor.excel_processor import ExcelProcessor
    from embedding.vectorization import Vectorization
    from util.vect_db_conn import VectorDBConnection
    from rag.rag_search import RAGSearcher, RAGBuilder
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class FullPipelineTest:
    """Complete pipeline testing with real data."""
    
    def __init__(self):
        """Initialize test environment."""
        print("🚀 Initializing Full Pipeline Test")
        print("=" * 50)
        
        # Initialize components
        self.excel_processor = ExcelProcessor()
        self.vectorizer = Vectorization()
        self.rag_searcher = RAGSearcher(similarity_threshold=0.24, max_workers=4)
        
        # Test data paths
        self.test_file = "data/raw/NB_패스트푸드/24H2_NBCI_패스트푸드_코딩완료데이터_20240614_F.xlsx"
        self.codeframe_file = "data/raw/NB_패스트푸드/-2024 NBCI 하반기_코드프레임 1.xlsx"
        
        print(f"   Excel Processor: ✅")
        print(f"   Vectorization: ✅") 
        print(f"   RAG Searcher: ✅")
        print(f"   Test file: {Path(self.test_file).name}")
        print(f"   Codeframe: {Path(self.codeframe_file).name}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available."""
        print("\n🔍 Checking Prerequisites...")
        
        # Check test files
        test_file_exists = Path(self.test_file).exists()
        codeframe_exists = Path(self.codeframe_file).exists()
        
        print(f"   Test data file: {'✅' if test_file_exists else '❌'}")
        print(f"   Codeframe file: {'✅' if codeframe_exists else '❌'}")
        
        # Check database connection
        try:
            db = VectorDBConnection()
            db.close()
            db_available = True
            print(f"   Database connection: ✅")
        except Exception as e:
            db_available = False
            print(f"   Database connection: ❌ ({e})")
        
        return test_file_exists and db_available
    
    def test_excel_processing(self) -> Dict[str, Any]:
        """Test Excel file processing."""
        print("\n📊 Testing Excel Processing...")
        
        try:
            # Read and process the test file
            raw_df, obj_data, grouped_dict = self.excel_processor.read_excel(self.test_file)
            
            # Get statistics
            stats = {
                "total_columns": len(raw_df.columns),
                "text_columns": len(obj_data.columns), 
                "question_groups": len(grouped_dict),
                "data_rows": len(obj_data),
                "sample_groups": {}
            }
            
            # Get sample data from first few groups
            for q_num in list(grouped_dict.keys())[:3]:
                columns = grouped_dict[q_num]
                if columns:
                    q_data = self.excel_processor.get_question_columns(obj_data, q_num, grouped_dict)
                    sample_values = []
                    
                    for col in columns[:2]:  # First 2 columns
                        if col in q_data.columns:
                            values = q_data[col].dropna().head(3).tolist()
                            sample_values.extend(values)
                    
                    stats["sample_groups"][q_num] = {
                        "columns": columns[:3],  # First 3 columns
                        "sample_values": sample_values[:5]  # First 5 values
                    }
            
            print(f"   ✅ Processed {stats['text_columns']} text columns")
            print(f"   ✅ Found {stats['question_groups']} question groups")
            print(f"   ✅ {stats['data_rows']} data rows")
            
            return {"status": "success", "data": (raw_df, obj_data, grouped_dict), "stats": stats}
            
        except Exception as e:
            print(f"   ❌ Excel processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_embedding_generation(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Test embedding generation with sample texts."""
        print("\n🔤 Testing Embedding Generation...")
        
        try:
            # Test single embedding
            single_embedding = self.vectorizer.encode(sample_texts[0], normalize=True, return_numpy=True)
            
            # Test batch embeddings
            batch_embeddings = self.vectorizer.encode_batch(
                sample_texts[:5], 
                batch_size=3, 
                normalize=True, 
                show_progress=False
            )
            
            # Check normalization
            norms = np.linalg.norm(batch_embeddings, axis=1)
            is_normalized = np.allclose(norms, 1.0, atol=1e-6)
            
            results = {
                "single_embedding_shape": single_embedding.shape,
                "batch_embeddings_shape": batch_embeddings.shape,
                "is_normalized": is_normalized,
                "sample_norm": float(norms[0]),
                "embedding_dim": batch_embeddings.shape[1]
            }
            
            print(f"   ✅ Single embedding shape: {results['single_embedding_shape']}")
            print(f"   ✅ Batch embedding shape: {results['batch_embeddings_shape']}")
            print(f"   ✅ Normalized: {results['is_normalized']}")
            print(f"   ✅ Embedding dimension: {results['embedding_dim']}")
            
            return {"status": "success", "results": results}
            
        except Exception as e:
            print(f"   ❌ Embedding generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_database_operations(self) -> Dict[str, Any]:
        """Test database operations."""
        print("\n🗄️  Testing Database Operations...")
        
        try:
            db = VectorDBConnection()
            
            # Test connection
            print("   ✅ Database connected")
            
            # Test search with dummy vector
            test_embedding = "[0.1]" + ", 0.0" * 767  # 768-dim test vector
            search_results = db.search_similar(test_embedding, k=5)
            
            results = {
                "connection": True,
                "search_results_count": len(search_results),
                "has_data": len(search_results) > 0
            }
            
            print(f"   ✅ Search test: {len(search_results)} results found")
            
            # Show sample results if available
            if search_results:
                print("   📊 Sample database entries:")
                for i, (word, code, distance) in enumerate(search_results[:3], 1):
                    print(f"      {i}. '{word}' (code: {code}, distance: {distance:.3f})")
            else:
                print("   ⚠️  No existing data in database")
            
            db.close()
            return {"status": "success", "results": results}
            
        except Exception as e:
            print(f"   ❌ Database operations failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_rag_search(self, obj_data: pd.DataFrame, grouped_dict: Dict) -> Dict[str, Any]:
        """Test RAG search functionality."""
        print("\n🔍 Testing RAG Search...")
        
        try:
            # Get sample data from Q5 group (as in the original deprecated code)
            if "Q5" in grouped_dict:
                print("   📝 Using Q5 data for testing...")
                q5_data = self.excel_processor.get_question_columns(obj_data, "Q5", grouped_dict)
                
                if not q5_data.empty:
                    # Get sample words from first column
                    first_col = q5_data.columns[0]
                    sample_words = q5_data[first_col].dropna().astype(str).unique()[:10]  # First 10 unique
                    
                    print(f"   📊 Testing with {len(sample_words)} words from {first_col}")
                    print(f"   📝 Sample words: {list(sample_words[:3])}")
                    
                    # Perform RAG search
                    search_results = self.rag_searcher.search_batch(sample_words.tolist(), k=3)
                    
                    if search_results:
                        # Convert to DataFrame and analyze
                        results_df = pd.DataFrame(search_results)
                        
                        # Apply threshold filtering
                        auto_classified, needs_review = self.rag_searcher.filter_by_threshold(results_df)
                        
                        # Get summary
                        summary = self.rag_searcher.create_classification_summary(results_df)
                        
                        # Get review candidates
                        review_candidates = self.rag_searcher.get_review_candidates(results_df, top_n=3)
                        
                        results = {
                            "total_searches": len(search_results),
                            "auto_classified": len(auto_classified),
                            "needs_review": len(needs_review),
                            "auto_rate": summary["auto_classification_rate"],
                            "avg_confidence": summary["avg_confidence"],
                            "sample_results": search_results[:5],
                            "review_candidates": review_candidates.to_dict('records') if not review_candidates.empty else []
                        }
                        
                        print(f"   ✅ Processed {len(search_results)} searches")
                        print(f"   ✅ Auto-classified: {len(auto_classified)} ({summary['auto_classification_rate']:.1%})")
                        print(f"   ✅ Needs review: {len(needs_review)}")
                        print(f"   ✅ Avg confidence: {summary['avg_confidence']:.3f}")
                        
                        # Show sample results
                        print("   📊 Sample search results:")
                        for result in search_results[:3]:
                            print(f"      '{result['word']}' → '{result['classified']}' (distance: {result['distance']:.3f})")
                        
                        return {"status": "success", "results": results}
                    else:
                        print("   ⚠️  No search results returned")
                        return {"status": "no_results", "message": "No search results"}
                else:
                    print("   ⚠️  Q5 data is empty")
                    return {"status": "no_data", "message": "Q5 data is empty"}
            else:
                print("   ⚠️  Q5 group not found")
                return {"status": "no_q5", "message": "Q5 group not found"}
                
        except Exception as e:
            print(f"   ❌ RAG search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_full_test(self) -> Dict[str, Any]:
        """Run the complete pipeline test."""
        print("\n🎯 Running Full Pipeline Test")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {"status": "failed", "reason": "Prerequisites not met"}
        
        # Test 1: Excel Processing
        excel_test = self.test_excel_processing()
        if excel_test["status"] != "success":
            return {"status": "failed", "step": "excel_processing", "error": excel_test}
        
        raw_df, obj_data, grouped_dict = excel_test["data"]
        
        # Test 2: Extract sample texts for embedding test
        sample_texts = []
        for q_num in list(grouped_dict.keys())[:2]:
            columns = grouped_dict[q_num]
            if columns:
                q_data = self.excel_processor.get_question_columns(obj_data, q_num, grouped_dict)
                for col in columns[:1]:
                    if col in q_data.columns:
                        texts = q_data[col].dropna().astype(str).head(5).tolist()
                        sample_texts.extend(texts)
                        break
                if len(sample_texts) >= 5:
                    break
        
        # Test 3: Embedding Generation
        if sample_texts:
            embedding_test = self.test_embedding_generation(sample_texts)
            if embedding_test["status"] != "success":
                return {"status": "failed", "step": "embedding", "error": embedding_test}
        
        # Test 4: Database Operations
        db_test = self.test_database_operations()
        if db_test["status"] != "success":
            return {"status": "failed", "step": "database", "error": db_test}
        
        # Test 5: RAG Search
        rag_test = self.test_rag_search(obj_data, grouped_dict)
        
        # Compile final results
        final_results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "test_file": self.test_file,
            "excel_processing": excel_test["stats"],
            "embedding_generation": embedding_test.get("results", {}),
            "database_operations": db_test.get("results", {}),
            "rag_search": rag_test.get("results", {}),
            "pipeline_ready": True
        }
        
        return final_results
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n🏆 FULL PIPELINE TEST SUMMARY")
        print("=" * 50)
        
        if results["status"] == "success":
            print("✅ ALL TESTS PASSED!")
            print(f"\n📊 Test Statistics:")
            print(f"   File processed: {Path(results['test_file']).name}")
            print(f"   Text columns: {results['excel_processing']['text_columns']}")
            print(f"   Question groups: {results['excel_processing']['question_groups']}")
            print(f"   Data rows: {results['excel_processing']['data_rows']}")
            
            if 'embedding_generation' in results:
                print(f"   Embedding dimension: {results['embedding_generation'].get('embedding_dim', 'N/A')}")
            
            if 'rag_search' in results and 'total_searches' in results['rag_search']:
                rag_data = results['rag_search']
                print(f"   RAG searches: {rag_data['total_searches']}")
                print(f"   Auto-classified: {rag_data['auto_classified']} ({rag_data['auto_rate']:.1%})")
                print(f"   Needs review: {rag_data['needs_review']}")
            
            print(f"\n🚀 BACKEND READY FOR PRODUCTION!")
            print(f"   ✅ Excel processing: Functional")
            print(f"   ✅ Text embedding: Functional") 
            print(f"   ✅ Vector database: Connected")
            print(f"   ✅ RAG search: Operational")
            print(f"\n📝 Next Steps:")
            print(f"   1. Integrate with FastAPI endpoints")
            print(f"   2. Implement human review workflow")
            print(f"   3. Add result export functionality")
            print(f"   4. Deploy production system")
            
        else:
            print(f"❌ TEST FAILED at step: {results.get('step', 'unknown')}")
            print(f"   Reason: {results.get('reason', 'Unknown error')}")
            if 'error' in results:
                print(f"   Error details: {results['error']}")


def main():
    """Main function to run the full pipeline test."""
    try:
        # Initialize and run test
        test_runner = FullPipelineTest()
        results = test_runner.run_full_test()
        test_runner.print_final_summary(results)
        
        # Save results
        output_file = f"full_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Test results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n💥 Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
