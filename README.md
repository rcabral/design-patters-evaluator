# Design Pattern Evaluator

A modular Python system for automating the evaluation of Large Language Models (LLMs) in detecting software design patterns. This tool supports multiple providers (OpenAI, Google, Ollama) and generates a comparative analysis CSV.

## Features

- **Multi-Provider Support**: OpenAI (GPT-4o), Google (Gemini 1.5 Pro), and Local LLMs via Ollama (Llama 3.1, Gemma 2).
- **Configurable Architecture**: Easily add models via `models_config.yaml`.
- **External Prompts**: System instructions are managed in `prompt_instruction.txt`.
- **Batch Processing**: Recursively scans directories and processes files in bulk.
- **Robust Parsing**: Extracts JSON from LLM responses even if mixed with conversational text.

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
├── design-patterns/       # Put your source code here
│   ├── singleton/         # Folder name = Ground Truth Label
│   │   ├── Database.java
│   │   └── Config.py
│   ├── observer/
│   │   └── Listener.cs
│   └── ...
├── main.py                # Main script
├── models_config.yaml     # Model config
├── prompt_instruction.txt # System prompt
├── .env                   # API Keys
└── results_analysis.csv   # Generated Output
```

## How to Run

1. Place your dataset in the `design-patterns` folder.
2. Run the script:
   ```bash
   python main.py
   ```
3. Check the console for progress.
4. Open `results_analysis.csv` to view the evaluation results.

## Output Format

The `results_analysis.csv` will contain:
- `filename`: Name of the analyzed file.
- `ground_truth_label`: The expected pattern (from folder name).
- `model_name`: The model used.
- `detected_pattern`: The pattern found by the LLM.
- `confidence`: Confidence score (0-100).
- `adherence`: Adherence score (0-100).
- `reasoning`: Explanation provided by the LLM.
- `raw_response`: Full text response for debugging.
