# Design Pattern Evaluator

A modular Python system for automating the evaluation of Large Language Models (LLMs) in detecting software design patterns. This tool supports multiple providers (OpenAI, Google, Ollama) and generates a comparative analysis CSV.

## Features

-   **Multi-Provider Support**: OpenAI (GPT-4o), Google (Gemini 1.5 Pro), and Cloud/Local LLMs via Ollama and GroqCloud (Llama 3.1, Gemma 2).
-   **Configurable Architecture**: Easily add models via `models_config.yaml`.
-   **External Prompts**: System instructions are managed in `prompt_instruction.txt`.
-   **Batch Processing**: Recursively scans directories and processes files in bulk.
-   **Robust Parsing**: Extracts JSON from LLM responses even if mixed with conversational text.

## Installation

1. **Clone the repository** (or navigate to the project folder).

2. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Clone the Datasets** (optional - only if you want to regenerate):

    ```bash
    cd datasets-raw
    git clone https://github.com/sdharren/Java-DPD-dataset.git
    git clone https://github.com/ptidejteam/ptidej-P-MARt.git
    cd ..
    ```

5. **Generate the Design Patterns Folder**:
    ```bash
    python datasets-raw/generate_datasets.py
    ```

## Configuration

### 1. Environment Variables

Create a `.env` file in the root directory to store your API keys. You can copy the example:

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```ini
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
```

### 2. Model Configuration

Edit `models_config.yaml` to define which models to use.

```yaml
models:
    - name: "gpt-5o"
      provider: "openai"
      api_key_env: "OPENAI_API_KEY"
      model_id: "gpt-5o"

    - name: "llama-3.1-local"
      provider: "ollama"
      endpoint_url: "http://localhost:11434/api/generate"
      model_id: "llama3.1"
```

### 3. System Prompt

Modify `prompt_instruction.txt` to change the instructions sent to the LLMs.

## Setup Local Models (Ollama)

To use local models like Llama 3.1 or Gemma 2, you need [Ollama](https://ollama.com/).

1. **Install Ollama**: Download from the official website.
2. **Pull Models**: Open your terminal and run:
    ```bash
    ollama pull llama3.1
    ollama pull gemma-3-4b-pt
    ```
3. **Start Server**: Ensure Ollama is running (usually runs in the background or via `ollama serve`).

## Directory Structure

The system expects the following structure:

```
project_root/
├── datasets-raw/                    # Raw datasets (optional)
│   ├── generate_datasets.py         # Dataset generator script
│   ├── Java-DPD-dataset/            # Cloned from GitHub
│   │   └── dataset/
│   ├── ptidej-P-MARt/               # Cloned from GitHub
│   └── P-MARt DPs v1.2/
│       └── Design Pattern List v1.2.xml
├── design-patterns/                 # Generated dataset
│   ├── singleton/
│   │   ├── DPD-*.java               # Files from Java-DPD-dataset
│   │   └── PMART-*.java             # Files from P-MARt
│   ├── factory/
│   ├── observer/
│   └── decorator/
├── results/                         # Evaluation results
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── results_analysis.csv     # Combined results
│       ├── metrics_summary.json     # Combined metrics
│       ├── metadata.json
│       ├── dpd/                     # DPD-only results
│       │   ├── results_analysis.csv
│       │   └── metrics_summary.json
│       └── pmart/                   # PMART-only results
│           ├── results_analysis.csv
│           └── metrics_summary.json
├── main.py                          # Main evaluation script
├── metrics.py                       # Metrics calculator
├── models_config.yaml               # Model configuration
├── prompt_instruction.txt           # System prompt
├── .env                             # API Keys
└── requirements.txt
```

## How to Run

1. **Generate the dataset** (first time only):

    ```bash
    python datasets-raw/generate_datasets.py
    ```

2. **Run the evaluation**:

    ```bash
    python main.py
    ```

3. **View results** in the `results/` folder with timestamp.

4. **Recalculate metrics** on existing results (optional):
    ```bash
    python metrics.py
    ```

## Output Format

The `results_analysis.csv` will contain:

-   `filename`: Name of the analyzed file.
-   `ground_truth_label`: The expected pattern (from folder name).
-   `model_name`: The model used.
-   `detected_pattern`: The pattern found by the LLM.
-   `confidence`: Confidence score (0-100).
-   `adherence`: Adherence score (0-100).
-   `reasoning`: Explanation provided by the LLM.
-   `raw_response`: Full text response for debugging.
