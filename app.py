"""
SEAL (Self-Adapting Language Models) - Streamlit Web Interface

A user-friendly web interface for the SEAL framework that provides easy access to:
- Few-shot learning on ARC-AGI tasks
- Knowledge incorporation from passages
- Model training, evaluation, and prediction
"""

import streamlit as st
import os
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
from datetime import datetime
import threading
import queue
import time

# Configure page
st.set_page_config(
    page_title="SEAL - Self-Adapting Language Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
if 'current_logs' not in st.session_state:
    st.session_state.current_logs = ""
if 'experiment_running' not in st.session_state:
    st.session_state.experiment_running = False

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("🤖 SEAL Navigation")
    page = st.sidebar.selectbox(
        "Choose Domain",
        ["Home", "Few-Shot (ARC-AGI)", "Knowledge Incorporation", "Experiments", "Results Dashboard"]
    )
    
    # Main content area
    if page == "Home":
        show_home_page()
    elif page == "Few-Shot (ARC-AGI)":
        show_few_shot_page()
    elif page == "Knowledge Incorporation":
        show_knowledge_page()
    elif page == "Experiments":
        show_experiments_page()
    elif page == "Results Dashboard":
        show_results_page()

def show_home_page():
    """Display the home page with overview and setup instructions"""
    
    st.title("🤖 SEAL - Self-Adapting Language Models")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to SEAL
        
        SEAL (**Se**lf-**A**dapting **L**LMs) is a framework for training language models via RL to generate self-edits 
        (finetuning data and other update directives for themselves) in response to new inputs.
        
        ### 🎯 Two Main Domains:
        
        **1. Few-Shot Learning (ARC-AGI)**
        - Adapt to new tasks from few-shot examples
        - Work with Abstract Reasoning Corpus (ARC) puzzles
        - Generate and evaluate self-edits for pattern recognition
        
        **2. Knowledge Incorporation**
        - Incorporate new factual knowledge from passages
        - Work with SQuAD-style question-answering tasks
        - Update model weights with new information
        """)
        
    with col2:
        st.markdown("### 🚀 Quick Start")
        
        # Environment check
        if check_environment():
            st.success("✅ Environment configured correctly")
        else:
            st.error("❌ Environment setup needed")
            
        st.markdown("""
        **Required Setup:**
        1. Python 3.12+ environment
        2. CUDA-compatible GPU
        3. OpenAI API key (optional)
        """)
        
        if st.button("🔧 Check System Requirements"):
            check_system_requirements()
    
    # Recent experiments
    st.markdown("---")
    st.subheader("📊 Recent Activity")
    
    if st.session_state.experiments:
        recent_experiments = pd.DataFrame(st.session_state.experiments[-5:])
        st.dataframe(recent_experiments, use_container_width=True)
    else:
        st.info("No experiments run yet. Start with the Few-Shot or Knowledge Incorporation tabs!")

def show_few_shot_page():
    """Display the few-shot learning interface"""
    
    st.title("🎯 Few-Shot Learning (ARC-AGI)")
    st.markdown("Adapt language models to new tasks using few-shot examples from ARC-AGI dataset")
    st.markdown("---")
    
    # Workflow selection
    workflow = st.selectbox(
        "Select Workflow",
        ["Self-Edit Training", "Model Evaluation", "RestEM Training", "Prediction/Inference"]
    )
    
    if workflow == "Self-Edit Training":
        show_self_edit_training()
    elif workflow == "Model Evaluation":
        show_model_evaluation()
    elif workflow == "RestEM Training":
        show_restem_training()
    elif workflow == "Prediction/Inference":
        show_prediction_interface()

def show_self_edit_training():
    """Interface for self-edit training workflow"""
    
    st.subheader("🔄 Self-Edit Training")
    st.markdown("Generate configuration suggestions and train LoRA adapters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        
        model_name = st.selectbox(
            "Base Model",
            [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct", 
                "Qwen/Qwen2.5-7B",
                "Custom Path"
            ]
        )
        
        if model_name == "Custom Path":
            model_name = st.text_input("Custom Model Path")
        
        experiment_name = st.text_input("Experiment Name", value=f"training_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Data files
        st.markdown("### Data Configuration")
        uploaded_challenge = st.file_uploader("Challenge File", type=['json'], key="challenge_upload")
        uploaded_solution = st.file_uploader("Solution File", type=['json'], key="solution_upload")
        
        # Use default files if none uploaded
        if not uploaded_challenge:
            challenge_file = "few-shot/data/arc-agi_training_challenges_filtered_1B_training_set.json"
        else:
            challenge_file = save_uploaded_file(uploaded_challenge, "challenges")
            
        if not uploaded_solution:
            solution_file = "few-shot/data/arc-agi_training_solutions_filtered_1B_training_set.json"  
        else:
            solution_file = save_uploaded_file(uploaded_solution, "solutions")
    
    with col2:
        st.markdown("### Training Parameters")
        
        n_tasks = st.number_input("Number of Tasks", min_value=1, max_value=50, value=12)
        n_self_edits = st.number_input("Self-Edits per Task", min_value=1, max_value=50, value=15)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        
        st.markdown("### LoRA Configuration")
        lora_rank = st.number_input("LoRA Rank", min_value=1, max_value=256, value=16)
        lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=256, value=16)
        
        st.markdown("### Augmentation Settings")
        use_basic_aug = st.checkbox("Basic Augmentations", value=True)
        use_size_aug = st.checkbox("Size Augmentations", value=False)
        use_chain_aug = st.checkbox("Chain Augmentations", value=False)
    
    # Advanced settings in expander
    with st.expander("🔧 Advanced Settings"):
        col3, col4 = st.columns(2)
        with col3:
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.0e")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=5)
        with col4:
            gradient_steps = st.number_input("Gradient Accumulation Steps", min_value=1, max_value=16, value=1)
            max_seq_length = st.number_input("Max Sequence Length", min_value=512, max_value=4096, value=2048)
    
    # Execute button
    if st.button("🚀 Start Self-Edit Training", type="primary", disabled=st.session_state.experiment_running):
        if not model_name:
            st.error("Please specify a model name")
            return
            
        # Build command
        cmd = [
            "python", "few-shot/self-edit.py",
            f"--experiment_name={experiment_name}",
            f"--challenge_file={challenge_file}",
            f"--solution_file={solution_file}",
            f"--model_name={model_name}",
            f"--n_tasks={n_tasks}",
            f"--n_self_edits_per_task={n_self_edits}",
            f"--temperature={temperature}",
            f"--lora_rank={lora_rank}",
            f"--lora_alpha={lora_alpha}",
            f"--learning_rate={learning_rate}",
            f"--per_device_train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={gradient_steps}"
        ]
        
        if use_basic_aug:
            cmd.append("--use_basic_augmentations")
        if use_size_aug:
            cmd.append("--use_size_augmentations")
        if use_chain_aug:
            cmd.append("--use_chain_augmentations")
        
        # Execute experiment
        execute_experiment("Self-Edit Training", cmd, experiment_name)

def show_model_evaluation():
    """Interface for model evaluation workflow"""
    
    st.subheader("📊 Model Evaluation")
    st.markdown("Evaluate trained LoRA adapters on ARC tasks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Selection")
        
        pretrained_checkpoint = st.selectbox(
            "Pretrained Checkpoint",
            [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "Custom Path"
            ]
        )
        
        if pretrained_checkpoint == "Custom Path":
            pretrained_checkpoint = st.text_input("Custom Checkpoint Path")
        
        # Experiment folder selection
        experiment_folder = st.text_input("Experiment Folder", placeholder="e.g., ~/tti/training_set_iteration_1")
        lora_folder = st.text_input("LoRA Checkpoints Folder", placeholder="e.g., ~/loras/self-edit/training_set_iteration_1")
        
        st.markdown("### Data Files")
        uploaded_data = st.file_uploader("Data File", type=['json'], key="eval_data_upload")
        uploaded_solutions = st.file_uploader("Solution File", type=['json'], key="eval_solution_upload")
        
    with col2:
        st.markdown("### Evaluation Parameters")
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        n_sample = st.number_input("Number of Samples", min_value=1, max_value=10, value=1)
        num_examples = st.number_input("Number of Examples", min_value=1, max_value=20, value=11)
        n_self_edits = st.number_input("Number of Self-Edits", min_value=1, max_value=50, value=15)
        
        st.markdown("### LoRA Settings")
        max_lora_rank = st.number_input("Max LoRA Rank", min_value=1, max_value=256, value=128)
        include_n = st.number_input("Include N", min_value=1, max_value=10, value=1)
        
        new_format = st.checkbox("Use New Format", value=True)
    
    if st.button("📊 Start Evaluation", type="primary", disabled=st.session_state.experiment_running):
        if not all([pretrained_checkpoint, experiment_folder, lora_folder]):
            st.error("Please fill in all required fields")
            return
        
        # Use default files if none uploaded
        data_file = save_uploaded_file(uploaded_data, "eval_data") if uploaded_data else "few-shot/data/arc-agi_training_challenges_filtered_1B_training_set.json"
        solution_file = save_uploaded_file(uploaded_solutions, "eval_solutions") if uploaded_solutions else "few-shot/data/arc-agi_training_solutions_filtered_1B_training_set.json"
        
        cmd = [
            "python", "few-shot/eval-self-edits.py",
            f"--experiment_folder={experiment_folder}",
            f"--pretrained_checkpoint={pretrained_checkpoint}",
            f"--lora_checkpoints_folder={lora_folder}",
            f"--temperature={temperature}",
            f"--n_sample={n_sample}",
            f"--data_file={data_file}",
            f"--solution_file={solution_file}",
            f"--max_lora_rank={max_lora_rank}",
            f"--include_n={include_n}",
            f"--num_examples={num_examples}",
            f"--n_self_edits={n_self_edits}"
        ]
        
        if new_format:
            cmd.append("--new_format")
        
        execute_experiment("Model Evaluation", cmd, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_restem_training():
    """Interface for RestEM training workflow"""
    
    st.subheader("🎓 RestEM Training")
    st.markdown("Behavioral cloning training on successful configurations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Configuration")
        
        configs_file = st.text_input("Configs and Indices File", placeholder="final_configs_and_indices.json")
        results_file = st.text_input("Results File", placeholder="final_results.json")
        
        model_name = st.selectbox(
            "Base Model",
            [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "Custom Path"
            ]
        )
        
        if model_name == "Custom Path":
            model_name = st.text_input("Custom Model Path")
    
    with col2:
        st.markdown("### Training Configuration")
        
        lora_rank = st.number_input("LoRA Rank", min_value=1, max_value=256, value=16)
        lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=256, value=16)
        num_epochs = st.number_input("Training Epochs", min_value=1, max_value=20, value=8)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=5)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.0e")
        gradient_steps = st.number_input("Gradient Accumulation Steps", min_value=1, max_value=16, value=1)
    
    if st.button("🎓 Start RestEM Training", type="primary", disabled=st.session_state.experiment_running):
        if not all([configs_file, results_file, model_name]):
            st.error("Please fill in all required fields")
            return
        
        cmd = [
            "python", "few-shot/BC-self-edit.py",
            f"--configs_and_indices={configs_file}",
            f"--results={results_file}",
            f"--model_name={model_name}",
            f"--lora_rank={lora_rank}",
            f"--lora_alpha={lora_alpha}",
            f"--num_train_epochs={num_epochs}",
            f"--per_device_train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={gradient_steps}",
            f"--learning_rate={learning_rate}"
        ]
        
        execute_experiment("RestEM Training", cmd, f"restem_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_prediction_interface():
    """Interface for prediction/inference"""
    
    st.subheader("🔮 Prediction & Inference")
    st.markdown("Generate predictions for ARC tasks using trained models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Selection")
        
        prediction_type = st.selectbox("Prediction Type", ["Baseline", "Custom Model"])
        
        if prediction_type == "Baseline":
            model_path = st.selectbox(
                "Baseline Model",
                [
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct"
                ]
            )
        else:
            model_path = st.text_input("Custom Model Path")
        
        st.markdown("### Input Data")
        uploaded_challenges = st.file_uploader("Challenge File", type=['json'], key="pred_challenge_upload")
        uploaded_solutions = st.file_uploader("Solution File (optional)", type=['json'], key="pred_solution_upload")
    
    with col2:
        st.markdown("### Generation Parameters")
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        n_sample = st.number_input("Number of Samples", min_value=1, max_value=10, value=1)
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=2048, value=512)
        
        if prediction_type == "Custom Model":
            num_examples = st.number_input("Number of Examples", min_value=1, max_value=20, value=9)
    
    if st.button("🔮 Generate Predictions", type="primary", disabled=st.session_state.experiment_running):
        if not model_path:
            st.error("Please specify a model path")
            return
        
        if not uploaded_challenges:
            st.error("Please upload a challenge file")
            return
        
        challenge_file = save_uploaded_file(uploaded_challenges, "pred_challenges")
        solution_file = save_uploaded_file(uploaded_solutions, "pred_solutions") if uploaded_solutions else None
        
        if prediction_type == "Baseline":
            script = "few-shot/predict_baseline.py"
        else:
            script = "few-shot/predict_custom.py"
        
        cmd = [
            "python", script,
            f"--model_path={model_path}",
            f"--challenge_file={challenge_file}",
            f"--temperature={temperature}",
            f"--n_sample={n_sample}",
            f"--max_tokens={max_tokens}"
        ]
        
        if solution_file:
            cmd.append(f"--solution_file={solution_file}")
        
        if prediction_type == "Custom Model":
            cmd.append(f"--num_examples={num_examples}")
        
        execute_experiment("Prediction", cmd, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_knowledge_page():
    """Display the knowledge incorporation interface"""
    
    st.title("📚 Knowledge Incorporation")
    st.markdown("Incorporate new factual knowledge from passages into language models")
    st.markdown("---")
    
    workflow = st.selectbox(
        "Select Workflow",
        ["Data Generation", "TTT Server Management", "Query/Evaluation", "Continual Learning", "SFT Training"]
    )
    
    if workflow == "Data Generation":
        show_data_generation()
    elif workflow == "TTT Server Management":
        show_ttt_server_management()
    elif workflow == "Query/Evaluation":
        show_query_evaluation()
    elif workflow == "Continual Learning":
        show_continual_learning()
    elif workflow == "SFT Training":
        show_sft_training()

def show_data_generation():
    """Interface for synthetic data generation"""
    
    st.subheader("🔨 Data Generation")
    st.markdown("Generate synthetic implications and completions from passages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Configuration")
        
        data_source = st.selectbox("Data Source", ["SQuAD Train", "SQuAD Val", "Custom Upload"])
        
        if data_source == "Custom Upload":
            uploaded_data = st.file_uploader("SQuAD-style JSON File", type=['json'])
        
        model_api = st.selectbox("Generation Model", ["OpenAI GPT-4", "OpenAI GPT-3.5", "Local Model"])
        
        if model_api.startswith("OpenAI"):
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        else:
            model_path = st.text_input("Local Model Path")
    
    with col2:
        st.markdown("### Generation Parameters")
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=1024, value=256)
        k_completions = st.number_input("Completions per Article", min_value=1, max_value=20, value=5)
        
        st.markdown("### Processing Options")
        num_articles = st.number_input("Number of Articles", min_value=1, max_value=1000, value=100)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=50, value=10)
    
    if st.button("🔨 Generate Data", type="primary", disabled=st.session_state.experiment_running):
        if model_api.startswith("OpenAI") and not api_key:
            st.error("Please provide OpenAI API key")
            return
        
        # Determine input file
        if data_source == "SQuAD Train":
            input_file = "knowledge-incorporation/data/squad_train.json"
        elif data_source == "SQuAD Val":
            input_file = "knowledge-incorporation/data/squad_val.json"
        else:
            if not uploaded_data:
                st.error("Please upload a data file")
                return
            input_file = save_uploaded_file(uploaded_data, "squad_custom")
        
        # Set up environment for OpenAI
        if model_api.startswith("OpenAI"):
            os.environ['OPENAI_API_KEY'] = api_key
        
        cmd = [
            "python", "knowledge-incorporation/src/data_generation/make_squad_data.py",
            f"--input_file={input_file}",
            f"--temperature={temperature}",
            f"--top_p={top_p}",
            f"--max_tokens={max_tokens}",
            f"--k={k_completions}",
            f"--num_articles={num_articles}",
            f"--batch_size={batch_size}"
        ]
        
        if model_api == "OpenAI GPT-3.5":
            cmd.append("--model=gpt-3.5-turbo")
        elif model_api == "Local Model":
            cmd.append(f"--model={model_path}")
        
        execute_experiment("Data Generation", cmd, f"datagen_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_ttt_server_management():
    """Interface for TTT server management"""
    
    st.subheader("🖥️ TTT Server Management")
    st.markdown("Manage Test-Time Training server for LoRA adaptation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Server Configuration")
        
        model_name = st.selectbox(
            "Base Model",
            [
                "Qwen/Qwen2.5-7B",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "Custom Path"
            ]
        )
        
        if model_name == "Custom Path":
            model_name = st.text_input("Custom Model Path")
        
        st.markdown("### Hardware Configuration")
        vllm_gpus = st.text_input("vLLM Server GPUs", value="0", help="Comma-separated GPU IDs")
        inner_loop_gpu = st.text_input("Inner Loop GPU", value="1", help="Single GPU ID")
        
        port = st.number_input("vLLM Port", min_value=8000, max_value=9999, value=8001)
        zmq_port = st.number_input("ZMQ Port", min_value=5000, max_value=6999, value=5555)
    
    with col2:
        st.markdown("### Model Parameters")
        
        max_seq_length = st.number_input("Max Sequence Length", min_value=512, max_value=4096, value=2048)
        eval_max_tokens = st.number_input("Eval Max Tokens", min_value=16, max_value=256, value=64)
        eval_temperature = st.slider("Eval Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        eval_top_p = st.slider("Eval Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        max_lora_rank = st.number_input("Max LoRA Rank", min_value=1, max_value=256, value=32)
    
    # Server status
    st.markdown("### Server Status")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("🚀 Start Server", type="primary"):
            start_ttt_server(model_name, vllm_gpus, inner_loop_gpu, port, zmq_port, 
                           max_seq_length, eval_max_tokens, eval_temperature, eval_top_p, max_lora_rank)
    
    with col4:
        if st.button("🛑 Stop Server", type="secondary"):
            stop_ttt_server()
    
    with col5:
        if st.button("🔍 Check Status"):
            check_server_status(port, zmq_port)

def show_query_evaluation():
    """Interface for querying the TTT server"""
    
    st.subheader("🔍 Query & Evaluation")
    st.markdown("Query TTT server for single or multi-passage evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Query Configuration")
        
        query_type = st.selectbox("Query Type", ["Single Passage", "Multi Passage (CPT)"])
        
        server_host = st.text_input("Server Host", value="localhost")
        zmq_port = st.number_input("ZMQ Port", min_value=5000, max_value=6999, value=5555)
        
        st.markdown("### Data Configuration")
        data_mode = st.selectbox("Data Mode", ["Training", "Validation"])
        uploaded_data = st.file_uploader("Data File (optional)", type=['json'])
        
        if query_type == "Multi Passage (CPT)":
            n_articles = st.number_input("Number of Articles", min_value=1, max_value=100, value=10)
    
    with col2:
        st.markdown("### Evaluation Parameters")
        
        n_sequences = st.number_input("Number of Sequences", min_value=1, max_value=50, value=5)
        n_datapoints = st.number_input("Number of Datapoints", min_value=1, max_value=100, value=20)
        
        st.markdown("### LoRA Configuration")
        lora_rank = st.number_input("LoRA Rank", min_value=1, max_value=256, value=32)
        lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=256, value=32)
        lora_dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        
        st.markdown("### Training Parameters")
        finetune_epochs = st.number_input("Finetune Epochs", min_value=1, max_value=20, value=3)
        finetune_lr = st.number_input("Finetune LR", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.0e")
    
    if st.button("🔍 Start Query", type="primary", disabled=st.session_state.experiment_running):
        if query_type == "Single Passage":
            script = "knowledge-incorporation/src/query/query_server.py"
        else:
            script = "knowledge-incorporation/src/query/CPT.py"
        
        cmd = [
            "python", script,
            f"--server_host={server_host}",
            f"--zmq_port={zmq_port}",
            f"--n_sequences={n_sequences}",
            f"--n_datapoints={n_datapoints}",
            f"--lora_rank={lora_rank}",
            f"--lora_alpha={lora_alpha}",
            f"--lora_dropout={lora_dropout}",
            f"--finetune_epochs={finetune_epochs}",
            f"--finetune_lr={finetune_lr}"
        ]
        
        if query_type == "Multi Passage (CPT)":
            cmd.append(f"--n_articles={n_articles}")
        
        if uploaded_data:
            data_file = save_uploaded_file(uploaded_data, "query_data")
            cmd.append(f"--data_file={data_file}")
        
        if data_mode == "Training":
            cmd.append("--training_mode")
        
        execute_experiment(f"{query_type} Query", cmd, f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_continual_learning():
    """Interface for continual learning experiments"""
    
    st.subheader("🔄 Continual Learning")
    st.markdown("Run continual self-edits experiments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Experiment Configuration")
        
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=20, value=8)
        documents_per_run = st.number_input("Documents per Run", min_value=1, max_value=20, value=5)
        
        server_host = st.text_input("Server Host", value="localhost")
        zmq_port = st.number_input("ZMQ Port", min_value=5000, max_value=6999, value=5555)
        
        st.markdown("### Data Configuration")
        uploaded_data = st.file_uploader("Training Data", type=['json'])
    
    with col2:
        st.markdown("### Learning Parameters")
        
        initial_epochs = st.number_input("Initial Training Epochs", min_value=1, max_value=20, value=5)
        continual_epochs = st.number_input("Continual Training Epochs", min_value=1, max_value=10, value=3)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.0e")
        
        st.markdown("### LoRA Configuration")
        lora_rank = st.number_input("LoRA Rank", min_value=1, max_value=256, value=32)
        lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=256, value=32)
    
    if st.button("🔄 Start Continual Learning", type="primary", disabled=st.session_state.experiment_running):
        cmd = [
            "python", "knowledge-incorporation/src/continual/continual_self_edits.py",
            f"--server_host={server_host}",
            f"--zmq_port={zmq_port}",
            f"--num_runs={num_runs}",
            f"--documents_per_run={documents_per_run}",
            f"--initial_epochs={initial_epochs}",
            f"--continual_epochs={continual_epochs}",
            f"--learning_rate={learning_rate}",
            f"--lora_rank={lora_rank}",
            f"--lora_alpha={lora_alpha}"
        ]
        
        if uploaded_data:
            data_file = save_uploaded_file(uploaded_data, "continual_data")
            cmd.append(f"--data_file={data_file}")
        
        execute_experiment("Continual Learning", cmd, f"continual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_sft_training():
    """Interface for supervised fine-tuning"""
    
    st.subheader("🎓 Supervised Fine-Tuning")
    st.markdown("Train models using supervised fine-tuning on generated datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        
        model_name = st.selectbox(
            "Base Model",
            [
                "Qwen/Qwen2.5-7B",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "Custom Path"
            ]
        )
        
        if model_name == "Custom Path":
            model_name = st.text_input("Custom Model Path")
        
        st.markdown("### Data Configuration")
        dataset_file = st.text_input("Dataset File", placeholder="path/to/sft_dataset.jsonl")
        uploaded_dataset = st.file_uploader("Upload Dataset", type=['json', 'jsonl'])
        
        output_dir = st.text_input("Output Directory", value="./sft_output")
    
    with col2:
        st.markdown("### Training Parameters")
        
        num_epochs = st.number_input("Training Epochs", min_value=1, max_value=20, value=3)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.0e")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
        gradient_steps = st.number_input("Gradient Accumulation Steps", min_value=1, max_value=16, value=4)
        
        st.markdown("### LoRA Configuration")
        lora_rank = st.number_input("LoRA Rank", min_value=1, max_value=256, value=16)
        lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=256, value=32)
        lora_dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    
    if st.button("🎓 Start SFT Training", type="primary", disabled=st.session_state.experiment_running):
        if not model_name:
            st.error("Please specify a model name")
            return
        
        # Use uploaded dataset if provided
        if uploaded_dataset:
            dataset_file = save_uploaded_file(uploaded_dataset, "sft_dataset")
        
        if not dataset_file:
            st.error("Please provide a dataset file")
            return
        
        cmd = [
            "python", "knowledge-incorporation/src/EM/train_SFT.py",
            f"--model_name={model_name}",
            f"--dataset_file={dataset_file}",
            f"--output_dir={output_dir}",
            f"--num_train_epochs={num_epochs}",
            f"--learning_rate={learning_rate}",
            f"--per_device_train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={gradient_steps}",
            f"--lora_rank={lora_rank}",
            f"--lora_alpha={lora_alpha}",
            f"--lora_dropout={lora_dropout}"
        ]
        
        execute_experiment("SFT Training", cmd, f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def show_experiments_page():
    """Display experiment monitoring and management"""
    
    st.title("🧪 Experiments")
    st.markdown("Monitor and manage running experiments")
    st.markdown("---")
    
    # Current experiment status
    if st.session_state.experiment_running:
        st.info("🔄 Experiment is currently running...")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Live Logs")
            log_container = st.container()
            with log_container:
                if st.session_state.current_logs:
                    st.code(st.session_state.current_logs, language="bash")
                else:
                    st.write("Waiting for logs...")
        
        with col2:
            if st.button("🛑 Stop Experiment", type="secondary"):
                st.session_state.experiment_running = False
                st.experimental_rerun()
    else:
        st.success("✅ No experiments currently running")
    
    # Experiment history
    st.markdown("### Experiment History")
    if st.session_state.experiments:
        df = pd.DataFrame(st.session_state.experiments)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.selectbox("Filter by Type", ["All"] + list(df['type'].unique()))
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(df['status'].unique()))
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.experiments = []
                st.experimental_rerun()
        
        # Apply filters
        filtered_df = df.copy()
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df['type'] == type_filter]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
        
        # Display table
        st.dataframe(
            filtered_df[['timestamp', 'name', 'type', 'status', 'duration']],
            use_container_width=True
        )
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Experiments", len(df))
        with col2:
            completed = len(df[df['status'] == 'Completed'])
            st.metric("Completed", completed)
        with col3:
            failed = len(df[df['status'] == 'Failed'])
            st.metric("Failed", failed)
        with col4:
            if len(df) > 0:
                success_rate = (completed / len(df)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
    else:
        st.info("No experiments have been run yet.")

def show_results_page():
    """Display results and visualizations"""
    
    st.title("📊 Results Dashboard")
    st.markdown("Visualize experiment results and model performance")
    st.markdown("---")
    
    if not st.session_state.experiments:
        st.info("No experiment results to display yet. Run some experiments first!")
        return
    
    # Performance over time
    df = pd.DataFrame(st.session_state.experiments)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Experiments Over Time")
        
        # Count experiments by date
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby(['date', 'type']).size().reset_index(name='count')
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(daily_counts, x='date', y='count', color='type', 
                        title="Experiments by Date and Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(daily_counts.set_index('date'))
    
    with col2:
        st.markdown("### Success Rate by Type")
        
        # Success rate by experiment type
        success_df = df.groupby('type').agg({
            'status': lambda x: (x == 'Completed').sum() / len(x) * 100
        }).reset_index()
        success_df.columns = ['type', 'success_rate']
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(success_df, x='type', y='success_rate',
                        title="Success Rate by Experiment Type")
            fig.update_layout(yaxis_title="Success Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(success_df.set_index('type'))
    
    # Detailed results
    st.markdown("### Detailed Results")
    
    # Load actual results if available
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        
        if result_files:
            selected_result = st.selectbox("Select Result File", result_files)
            
            if selected_result:
                try:
                    with open(selected_result, 'r') as f:
                        result_data = json.load(f)
                    
                    # Display results based on type
                    if "accuracy" in result_data:
                        st.metric("Accuracy", f"{result_data['accuracy']:.3f}")
                    
                    if "loss" in result_data:
                        st.metric("Loss", f"{result_data['loss']:.3f}")
                    
                    # Plot training curves if available
                    if "training_history" in result_data:
                        history = result_data["training_history"]
                        
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=history.get("loss", []),
                                name="Training Loss",
                                line=dict(color="red")
                            ))
                            
                            if "val_loss" in history:
                                fig.add_trace(go.Scatter(
                                    y=history["val_loss"],
                                    name="Validation Loss",
                                    line=dict(color="blue")
                                ))
                            
                            fig.update_layout(title="Training Progress", 
                                            xaxis_title="Epoch", 
                                            yaxis_title="Loss")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to simple line chart
                            chart_data = pd.DataFrame()
                            if "loss" in history:
                                chart_data["Training Loss"] = history["loss"]
                            if "val_loss" in history:
                                chart_data["Validation Loss"] = history["val_loss"]
                            st.line_chart(chart_data)
                    
                    # Show raw data
                    with st.expander("📋 Raw Result Data"):
                        st.json(result_data)
                        
                except Exception as e:
                    st.error(f"Error loading result file: {e}")
        else:
            st.info("No result files found in results/ directory")
    else:
        st.info("Results directory not found")

# Utility functions

def check_environment():
    """Check if the environment is properly configured"""
    try:
        import torch
        import transformers
        import accelerate
        import peft
        
        # Check for GPU
        gpu_available = torch.cuda.is_available()
        
        # Check for OpenAI key
        openai_key = os.getenv('OPENAI_API_KEY')
        
        return gpu_available
    except ImportError:
        return False

def check_system_requirements():
    """Display system requirements check"""
    requirements = {
        "Python Version": sys.version_info >= (3, 12),
        "PyTorch": False,
        "Transformers": False,
        "Accelerate": False,
        "PEFT": False,
        "CUDA": False,
        "OpenAI API Key": bool(os.getenv('OPENAI_API_KEY'))
    }
    
    try:
        import torch
        requirements["PyTorch"] = True
        requirements["CUDA"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import transformers
        requirements["Transformers"] = True
    except ImportError:
        pass
    
    try:
        import accelerate
        requirements["Accelerate"] = True
    except ImportError:
        pass
    
    try:
        import peft
        requirements["PEFT"] = True
    except ImportError:
        pass
    
    # Display results
    for req, status in requirements.items():
        if status:
            st.success(f"✅ {req}")
        else:
            st.error(f"❌ {req}")

def save_uploaded_file(uploaded_file, prefix):
    """Save uploaded file and return path"""
    if uploaded_file is None:
        return None
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = uploads_dir / f"{prefix}_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def execute_experiment(exp_type, cmd, exp_name):
    """Execute experiment and track progress"""
    st.session_state.experiment_running = True
    st.session_state.current_logs = ""
    
    # Add experiment to history
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "name": exp_name,
        "type": exp_type,
        "command": " ".join(cmd),
        "status": "Running",
        "duration": "0:00:00"
    }
    
    start_time = datetime.now()
    
    try:
        # Use st.empty() containers for live updates
        progress_placeholder = st.empty()
        log_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info(f"🚀 Starting {exp_type}: {exp_name}")
        
        # Execute command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        logs = []
        for line in process.stdout:
            logs.append(line.rstrip())
            st.session_state.current_logs = "\n".join(logs[-50:])  # Keep last 50 lines
            
            with log_placeholder.container():
                st.code(st.session_state.current_logs, language="bash")
        
        # Wait for completion
        return_code = process.wait()
        
        # Update experiment status
        end_time = datetime.now()
        duration = end_time - start_time
        
        experiment["status"] = "Completed" if return_code == 0 else "Failed"
        experiment["duration"] = str(duration).split('.')[0]  # Remove microseconds
        
        with progress_placeholder.container():
            if return_code == 0:
                st.success(f"✅ {exp_type} completed successfully!")
            else:
                st.error(f"❌ {exp_type} failed with return code {return_code}")
        
    except Exception as e:
        experiment["status"] = "Failed"
        experiment["duration"] = str(datetime.now() - start_time).split('.')[0]
        
        with progress_placeholder.container():
            st.error(f"❌ Error running {exp_type}: {str(e)}")
    
    finally:
        st.session_state.experiments.append(experiment)
        st.session_state.experiment_running = False

def start_ttt_server(model_name, vllm_gpus, inner_loop_gpu, port, zmq_port, 
                    max_seq_length, eval_max_tokens, eval_temperature, eval_top_p, max_lora_rank):
    """Start TTT server"""
    
    # Build server command
    cmd = [
        "bash", "knowledge-incorporation/scripts/TTT_server.sh"
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        "MODEL_NAME": model_name,
        "VLLM_SERVER_GPUS": vllm_gpus,
        "INNER_LOOP_GPU": inner_loop_gpu,
        "PORT": str(port),
        "ZMQ_PORT": str(zmq_port),
        "MAX_SEQ_LENGTH": str(max_seq_length),
        "EVAL_MAX_TOKENS": str(eval_max_tokens),
        "EVAL_TEMPERATURE": str(eval_temperature),
        "EVAL_TOP_P": str(eval_top_p),
        "MAX_LORA_RANK": str(max_lora_rank)
    })
    
    try:
        # Start server in background
        process = subprocess.Popen(cmd, env=env)
        st.success(f"🚀 TTT Server starting with PID {process.pid}")
        st.info(f"Check logs for status. Server will be available on port {port} (vLLM) and {zmq_port} (ZMQ)")
    except Exception as e:
        st.error(f"Failed to start TTT server: {e}")

def stop_ttt_server():
    """Stop TTT server"""
    try:
        # Kill processes using the ports
        subprocess.run(["pkill", "-f", "TTT_server"], check=False)
        subprocess.run(["pkill", "-f", "vllm"], check=False)
        st.success("🛑 TTT Server stopped")
    except Exception as e:
        st.error(f"Error stopping server: {e}")

def check_server_status(port, zmq_port):
    """Check TTT server status"""
    import requests
    import socket
    
    # Check vLLM server
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        vllm_status = "🟢 Running" if response.status_code == 200 else "🔴 Error"
    except:
        vllm_status = "🔴 Not responding"
    
    # Check ZMQ port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', zmq_port))
        zmq_status = "🟢 Open" if result == 0 else "🔴 Closed"
        sock.close()
    except:
        zmq_status = "🔴 Error"
    
    st.write(f"**vLLM Server (port {port}):** {vllm_status}")
    st.write(f"**ZMQ Server (port {zmq_port}):** {zmq_status}")

if __name__ == "__main__":
    main()