import time
import sys
import os
import subprocess
import requests

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from a2a.client import send_task, send_task_streaming
from a2a.schemas import A2ATask
from core.hitl import ask_permission
from core.file_writer import write_project
from core.user_input import get_user_input
from config import SERVICE_URL, SERVICE_PORT, MAX_OPTIMIZATION_ITERATIONS, TARGET_ACCURACY
from core.trace_logger import PipelineTracer

OUT = os.path.join(PROJECT_ROOT, f"autoeda_output_{int(time.time())}")

# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

def check_service_running():
    """Check if unified service is already running"""
    try:
        response = requests.get(SERVICE_URL, timeout=1)
        return response.status_code == 200
    except Exception:
        return False

def start_service():
    """Start the unified service"""
    print("🚀 Starting AutoEDA Unified Service...")
    
    # Build command with proper module path
    service_file = os.path.join(PROJECT_ROOT, "unified_service.py")
    
    # Check if file exists
    if not os.path.exists(service_file):
        print(f"❌ unified_service.py not found at {service_file}")
        return None
    
    service_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "unified_service:app", "--port", str(SERVICE_PORT)],
        cwd=PROJECT_ROOT,  # Run from project root
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT}  # Add to Python path
    )
    
    # Wait for service to be ready
    max_retries = 30
    for i in range(max_retries):
        if check_service_running():
            print("✅ Service is ready!\n")
            return service_process
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"   Waiting for service... ({i+1}/{max_retries})")
    
    # Service failed - show error output
    print("❌ Failed to start service\n")
    print("📋 Error output:")
    stderr_output = service_process.stderr.read().decode('utf-8', errors='ignore')
    if stderr_output:
        print(stderr_output[:1000])  # Show first 1000 chars
    else:
        print("  No error output captured")
    
    service_process.terminate()
    return None

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_workflow():
    """Run the AutoML workflow"""
    print("=" * 70)
    print("  AutoEDA - Smart AutoML Workflow")
    print("=" * 70)

    # Get user input for CSV path and target column
    csv_path, target_column, _ = get_user_input()

    # Initialize pipeline tracer
    tracer = PipelineTracer(csv_path, target_column)

    # Step 1: Analysis
    print("\n📊 Step 1/7: Analyzing dataset...")
    analysis = send_task_streaming(
        f"{SERVICE_URL}/a2a/analysis",
        A2ATask.create(
            "orchestrator", "analysis-agent",
            "dataset_analysis", {"csv_path": csv_path, "target_column": target_column, "output_folder": OUT}
        )
    )
    print("✅ Analysis complete!")
    tracer.record("analysis", analysis["output"])

    # Step 2: Generate Insights (AI) — uses GROQ_API_KEY_1
    requests.post(f"{SERVICE_URL}/swap-key/1")
    print("\n💡 Step 2/7: Generating insights (AI)...")
    insights = send_task_streaming(
        f"{SERVICE_URL}/a2a/insight",
        A2ATask.create(
            "orchestrator", "insight-agent",
            "generate_insights", analysis["output"]
        )
    )
    print("✅ Insights generated!")
    tracer.record("insights_1", insights["output"])

    # Step 3: Preprocessing Strategy — uses GROQ_API_KEY_2
    requests.post(f"{SERVICE_URL}/swap-key/2")
    print("\n🧹 Step 3/7: Determining preprocessing strategy (AI)...")
    prep_strategy = send_task_streaming(
        f"{SERVICE_URL}/a2a/preprocessing",
        A2ATask.create(
            "orchestrator", "preprocessing-agent",
            "preprocessing_strategy", analysis["output"]
        )
    )
    print("✅ Preprocessing strategy determined!")
    tracer.record("preprocessing", prep_strategy["output"])

    # Step 4: Feature Engineering — uses GROQ_API_KEY_2 (same key)
    print("\n🔧 Step 4/7: Feature engineering strategy (AI)...")
    feat_strategy = send_task_streaming(
        f"{SERVICE_URL}/a2a/feature",
        A2ATask.create(
            "orchestrator", "feature-agent",
            "feature_engineering", {
                **analysis["output"],
                **prep_strategy["output"]
            }
        )
    )
    print("✅ Feature engineering strategy ready!")
    tracer.record("feature", feat_strategy["output"])

    # Step 5: Model Training
    print("\n🤖 Step 5/7: Training models...")
    print("  ⏳ Smart ML active: Tuning, SMOTE, CV & SHAP may take a few minutes...")
    
    # Train models (single pass — internal CV, SMOTE, and Optuna handle optimization)
    models = send_task_streaming(
        f"{SERVICE_URL}/a2a/model",
        A2ATask.create(
            "orchestrator", "model-agent",
            "model_training", {
                "csv_path": csv_path,
                "target_column": target_column,
                "prep_strategy": prep_strategy["output"],
                "feat_strategy": feat_strategy["output"],
                "output_folder": OUT
            }
        )
    )
    
    print(f"  Best model: {models['output']['best_model']}")
    print(f"  Score: {models['output']['best_score']:.4f}")
    
    if models['output']['best_score'] >= TARGET_ACCURACY:
        print(f"  ✅ Target accuracy ({TARGET_ACCURACY}) achieved!")
    else:
        print(f"  ⚠️  Below target ({TARGET_ACCURACY}), but internal Optuna tuning already optimized.")
    
    # Evaluate
    evaluation = send_task_streaming(
        f"{SERVICE_URL}/a2a/evaluation",
        A2ATask.create(
            "orchestrator", "evaluation-agent",
            "model_evaluation", models["output"]
        )
    )

    print("\n✅ Model training complete!")
    tracer.record("models", models["output"])
    tracer.record("evaluation", evaluation["output"])

    # Swap to GROQ_API_KEY_3 for remaining LLM calls
    requests.post(f"{SERVICE_URL}/swap-key/3")

    # Step 6: Generate insights — uses GROQ_API_KEY_3
    print("\n💡 Step 6/7: Generating insights (AI)...")
    insights = send_task_streaming(
        f"{SERVICE_URL}/a2a/insight",
        A2ATask.create(
            "orchestrator", "insight-agent",
            "insight_generation", {
                **analysis["output"],
                **models["output"]
            }
        )
    )
    print("✅ Insights generated!")
    tracer.record("insights_2", insights["output"])

    # Step 7: Generate Project
    print("\n📝 Step 7/7: Generating project code (AI)...")
    project = send_task_streaming(
        f"{SERVICE_URL}/a2a/project",
        A2ATask.create(
            "orchestrator", "project-agent",
            "project_generation",
            {
                "analysis_summary": analysis["output"],
                "best_model_info": {
                    "model": models["output"]["best_model"],
                    "score": models["output"]["best_score"],
                    "problem_type": models["output"]["problem_type"]
                }
            }
        )
    )
    print("✅ Project code generated!")
    tracer.record("project", project["output"])

    # Generate trace report
    trace_report = tracer.generate_report()

    # Ask permission and write
    print("\n" + "=" * 70)
    if ask_permission(OUT):
        write_project(
            OUT,
            csv_path,
            project["output"]["analysis_code"],
            project["output"]["readme"],
            insights["output"]["insights"],
            trace_report
        )
        print(f"\n🎉 AutoML project created at: {OUT}")
        print(f"   📋 Pipeline trace saved to: {OUT}/pipeline_trace.md")
        print(f"   Best Model: {models['output']['best_model']}")
        print(f"   Score: {models['output']['best_score']:.4f}")
    print("=" * 70)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    service_process = None
    service_was_running = check_service_running()
    
    try:
        # Start service if not already running
        if not service_was_running:
            service_process = start_service()
            if not service_process:
                print("\n❌ Could not start service. Exiting.")
                sys.exit(1)
        else:
            print("✅ Service already running!\n")
        
        # Run the workflow
        run_workflow()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up: only stop service if we started it
        if service_process and not service_was_running:
            print("\n🛑 Stopping service...")
            service_process.terminate()
            service_process.wait(timeout=5)
            print("✅ Service stopped")
        
        print("\n👋 Goodbye!\n")