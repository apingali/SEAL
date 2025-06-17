# 📚 Knowledge Incorporation Module - Complete Guide

## Overview

The Knowledge Incorporation module enables language models to dynamically learn and integrate new factual knowledge from text passages. Unlike traditional fine-tuning that requires large datasets and compute, this approach allows models to quickly adapt to new information through self-generated training data.

---

## 🔄 Core Workflows Explained

### 1. Data Generation Workflow
**Purpose**: Convert raw text passages into synthetic training data

**How it works**:
- Takes passages with questions/answers (SQuAD format)
- Uses LLMs (GPT-4, local models) to generate:
  - **Implications**: What the passage implies beyond stated facts
  - **Completions**: Natural continuations of the text
  - **Self-QA**: Additional question-answer pairs

**Example Input** (SQuAD format):
```json
{
  "title": "Climate Change",
  "context": "Global temperatures have risen by 1.1°C since 1880...",
  "questions": [
    {
      "question": "How much have global temperatures risen?",
      "answers": [{"text": "1.1°C since 1880"}]
    }
  ]
}
```

**Generated Output**:
```json
{
  "implications": [
    "This warming trend suggests accelerating climate impacts",
    "Pre-industrial baselines are crucial for measuring change"
  ],
  "completions": [
    "...leading to more frequent extreme weather events",
    "...requiring immediate mitigation strategies"
  ],
  "synthetic_qa": [
    {
      "question": "What does the 1.1°C increase indicate?",
      "answer": "Significant global warming since the industrial era"
    }
  ]
}
```

### 2. TTT (Test-Time Training) Server
**Purpose**: Provides on-demand model adaptation infrastructure

**Architecture**:
- **vLLM Server**: Fast inference with LoRA support
- **ZMQ Server**: Handles training requests
- **GPU Management**: Separates inference and training workloads

**Workflow**:
1. Receives passage + questions
2. Generates synthetic training data
3. Trains temporary LoRA adapter
4. Evaluates on questions
5. Returns performance metrics

**Use Case**: When you need to quickly adapt a model to domain-specific knowledge without full retraining.

### 3. Query/Evaluation Workflows

#### Single Passage Query
- Tests model adaptation on individual documents
- Measures knowledge retention and application
- Useful for document-specific tasks

#### Multi-Passage (CPT - Continued Pre-Training)
- Processes multiple related documents
- Builds cumulative knowledge across passages
- Better for domain adaptation

**Example Flow**:
```
Input: Medical research papers → 
Generate training data → 
Train LoRA adapter → 
Test on medical Q&A → 
Measure accuracy improvement
```

### 4. Continual Learning
**Purpose**: Progressive knowledge accumulation across multiple documents

**Process**:
1. Start with base model
2. Learn from Document 1 → Update weights
3. Learn from Document 2 → Update weights (retaining previous knowledge)
4. Continue sequentially
5. Evaluate knowledge retention across all documents

**Key Challenge**: Avoiding catastrophic forgetting while learning new information.

### 5. SFT (Supervised Fine-Tuning)
**Purpose**: Traditional fine-tuning on generated datasets

**When to use**:
- When you have large amounts of generated data
- For permanent model updates
- When computational resources allow full training

---

## 📊 Data Sources & Formats

### Primary Sources
1. **SQuAD Dataset**: Question-answering on Wikipedia passages
2. **Custom Uploads**: Your own passage + Q&A data
3. **Generated Synthetic Data**: AI-created training examples

### Required Format
```json
{
  "data": [
    {
      "title": "Document Title",
      "paragraphs": [
        {
          "context": "Your text passage here...",
          "qas": [
            {
              "question": "What is...?",
              "answers": [
                {
                  "text": "Answer text",
                  "answer_start": 123
                }
              ],
              "id": "unique_id"
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 🚀 Practical Applications & Examples

### 1. Legal Document Analysis
**Scenario**: Law firm needs to quickly understand new regulations

**Implementation**:
```python
# Upload new regulation documents
# Generate implications and legal interpretations
# Train model on legal reasoning
# Query: "What are the compliance requirements for X?"
```

**Workflow**:
- **Data Generation**: Convert regulations into Q&A format
- **TTT Server**: Quick adaptation to legal domain
- **Query**: Test understanding of specific clauses
- **Result**: Model can answer complex legal questions

### 2. Medical Literature Integration
**Scenario**: Healthcare system needs to incorporate latest research

**Example Data**:
```json
{
  "title": "COVID-19 Treatment Guidelines",
  "context": "Recent studies show that monoclonal antibodies reduce hospitalization by 70% when administered within 72 hours...",
  "questions": [
    {
      "question": "When should monoclonal antibodies be administered?",
      "answers": [{"text": "within 72 hours"}]
    }
  ]
}
```

**Implementation Steps**:
1. **Data Generation**: Medical papers → Treatment protocols
2. **Continual Learning**: Accumulate knowledge from multiple studies
3. **Evaluation**: Test on clinical scenarios
4. **Deployment**: Model assists with treatment decisions

### 3. Corporate Knowledge Management
**Scenario**: Company needs to onboard employees with internal knowledge

**Use Case**:
- Upload company policies, procedures, product docs
- Generate comprehensive Q&A datasets
- Train model on company-specific knowledge
- Deploy as internal chatbot

**Example Workflow**:
```
Internal Docs → Knowledge Extraction → Model Training → Employee Q&A System
```

### 4. Educational Content Adaptation
**Scenario**: Adaptive learning system for different subjects

**Implementation**:
- **Math**: Upload textbook chapters → Generate problem variations
- **History**: Historical documents → Generate timeline questions
- **Science**: Research papers → Generate concept explanations

### 5. Financial Analysis
**Scenario**: Investment firm analyzing market reports

**Data Flow**:
```json
{
  "title": "Q3 Earnings Report - TechCorp",
  "context": "Revenue increased 15% YoY to $2.1B, driven by cloud services growth...",
  "generated_implications": [
    "Strong cloud adoption indicates sustainable growth",
    "15% growth rate suggests market leadership position"
  ]
}
```

**Applications**:
- Earnings analysis automation
- Risk assessment from news articles
- Market trend identification

### 6. News & Current Events
**Scenario**: News organization needs real-time fact integration

**Workflow**:
1. **Real-time ingestion**: Breaking news articles
2. **Knowledge extraction**: Key facts and implications
3. **Model updates**: Incorporate latest information
4. **Query system**: Answer questions about current events

---

## ⚙️ Configuration Examples

### Quick Medical Domain Adaptation
```python
# Data Generation Settings
temperature = 0.7  # Creative but focused
k_completions = 8  # Multiple perspectives
model = "gpt-4"   # High-quality generation

# TTT Server Settings
max_seq_length = 2048     # Handle long medical texts
eval_max_tokens = 128     # Detailed answers
lora_rank = 32           # Good adaptation capacity

# Training Settings
finetune_epochs = 5       # Quick adaptation
finetune_lr = 1e-4       # Stable learning
```

### Legal Document Processing
```python
# Focused on precision and consistency
temperature = 0.3        # Conservative generation
n_articles = 50         # Comprehensive coverage
lora_rank = 64          # Higher capacity for complex reasoning
finetune_epochs = 8     # Thorough learning
```

### Corporate Knowledge Base
```python
# Balanced for broad coverage
temperature = 0.5       # Moderate creativity
continual_epochs = 3    # Quick updates
documents_per_run = 10  # Batch processing
```

---

## 🎯 Key Benefits

1. **Rapid Adaptation**: Minutes vs. hours for traditional fine-tuning
2. **Resource Efficient**: LoRA adapters require minimal GPU memory
3. **Scalable**: Handle new documents continuously
4. **Domain Agnostic**: Works across any text domain
5. **Cost Effective**: No need for massive labeled datasets

---

## 📋 Step-by-Step Usage Guide

### Getting Started with Knowledge Incorporation

#### Step 1: Prepare Your Data
1. Format your documents in SQuAD-style JSON
2. Ensure each document has context and questions
3. Upload via Streamlit interface or save to `knowledge-incorporation/data/`

#### Step 2: Generate Synthetic Data
1. Navigate to "Data Generation" in the Streamlit app
2. Select your data source (SQuAD Train/Val or Custom Upload)
3. Configure generation parameters:
   - **Temperature**: 0.3-0.7 for factual content
   - **Top P**: 0.8-1.0 for diversity
   - **K Completions**: 3-8 per article
4. Click "Generate Data" and monitor progress

#### Step 3: Start TTT Server
1. Go to "TTT Server Management"
2. Configure hardware settings:
   - **vLLM GPUs**: GPUs for inference (e.g., "0,1")
   - **Inner Loop GPU**: GPU for training (e.g., "2")
   - **Ports**: Ensure no conflicts (default: 8001, 5555)
3. Set model parameters based on your use case
4. Click "Start Server" and wait for initialization

#### Step 4: Run Experiments
1. Choose between "Single Passage" or "Multi Passage (CPT)"
2. Configure LoRA parameters:
   - **Rank**: 16-64 (higher for complex domains)
   - **Alpha**: Same as rank for balanced adaptation
   - **Dropout**: 0.05-0.1 for regularization
3. Set training parameters:
   - **Epochs**: 3-8 (more for complex domains)
   - **Learning Rate**: 1e-5 to 1e-3
4. Start query and monitor results

#### Step 5: Analyze Results
1. Check "Results Dashboard" for performance metrics
2. Compare accuracy before/after adaptation
3. Evaluate knowledge retention across documents
4. Export results for further analysis

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### TTT Server Won't Start
- **Check GPU availability**: `nvidia-smi`
- **Verify ports are free**: `netstat -tlnp | grep :8001`
- **Ensure sufficient GPU memory**: Models require 8GB+ VRAM
- **Check .env file**: Ensure OPENAI_API_KEY is set if using OpenAI

#### Poor Adaptation Performance
- **Increase LoRA rank**: Try 32-64 for complex domains
- **Adjust learning rate**: Lower (1e-5) for stability, higher (1e-3) for faster learning
- **More training epochs**: 5-10 for thorough adaptation
- **Better synthetic data**: Lower temperature (0.3) for factual accuracy

#### Memory Issues
- **Reduce batch size**: Lower per_device_train_batch_size
- **Shorter sequences**: Reduce max_seq_length to 1024-1536
- **Gradient accumulation**: Increase steps to maintain effective batch size
- **Model sharding**: Use smaller base models (1B-3B parameters)

#### Data Quality Problems
- **Review generated examples**: Check for hallucinations or irrelevant content
- **Adjust generation parameters**: Lower temperature, higher top_p
- **Filter low-quality data**: Remove examples with low confidence scores
- **Domain-specific prompts**: Customize generation prompts for your domain

---

## 📚 Advanced Usage Patterns

### Domain-Specific Adaptation Pipeline
```python
# 1. Collect domain documents
medical_papers = load_medical_literature()

# 2. Generate high-quality synthetic data
synthetic_data = generate_data(
    papers=medical_papers,
    temperature=0.4,  # Conservative for medical facts
    model="gpt-4",    # High quality
    k_completions=5
)

# 3. Progressive learning
for batch in chunk_documents(medical_papers, size=10):
    results = continual_learning(
        documents=batch,
        lora_rank=64,     # High capacity
        epochs=8,         # Thorough learning
        eval_questions=medical_qa_test
    )
    track_performance(results)
```

### Real-Time Knowledge Updates
```python
# Monitor for new documents
while True:
    new_docs = check_for_updates()
    if new_docs:
        # Quick adaptation
        adapt_model(
            documents=new_docs,
            epochs=3,           # Fast updates
            retain_previous=True  # Avoid forgetting
        )
        # Validate performance
        test_knowledge_retention()
    sleep(3600)  # Check hourly
```

### Multi-Domain Knowledge Base
```python
# Separate adapters for different domains
domains = {
    "legal": create_lora_adapter(rank=32),
    "medical": create_lora_adapter(rank=64),
    "financial": create_lora_adapter(rank=32)
}

# Route queries to appropriate domain
def query_knowledge(question, domain):
    adapter = domains[domain]
    return model.generate(question, lora_adapter=adapter)
```

---

The Knowledge Incorporation module transforms any text into a learning opportunity, enabling models to stay current with new information while maintaining efficiency and accuracy. This approach democratizes knowledge integration, making it accessible without requiring extensive ML expertise or computational resources.