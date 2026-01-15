

# Financial Document Summarization & Evaluation Pipeline

A high-performance machine learning pipeline designed to automate the summarization and quality evaluation of complex financial documents, such as SEC filings, using state-of-the-art NLP models.

## 🚀 Project Overview

This project provides a robust workflow for processing financial texts. It leverages the **BART** (Bidirectional and Auto-Regressive Transformers) model to generate abstractive summaries and includes a full evaluation suite using **ROUGE** metrics.

### Key Features

* **Abstractive Summarization**: Uses pre-trained BART models specifically optimized for long-form financial text.
* **Performance Metrics**: Automatically calculates **ROUGE-1, ROUGE-2, and ROUGE-L** scores to validate summary accuracy.
* **Automated Reporting**: Generates visual performance charts (PNG) and detailed metric logs (CSV/TXT).
* **Fine-Tuning Support**: Includes integrated logic to fine-tune models on custom financial datasets.
* **Multi-Platform Ready**: Optimized for Google Colab (with GPU acceleration), Kaggle, and local environments.

## 🛠️ Setup & Installation

### 1. Google Colab (Recommended)

1. Upload the `Code File.ipynb` to your Colab environment.
2. Enable GPU acceleration: **Runtime** -> **Change runtime type** -> **GPU** (e.g., T4).
3. Run the first cell to execute `setup_colab_environment()` to mount Google Drive for persistent storage.

### 2. Kaggle Setup

1. Create a new notebook and copy the provided code.
2. Enable **GPU/TPU** in the notebook settings.
3. Execute the `main()` function.

### 3. Local Setup

Ensure you have a GPU with CUDA support installed.

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the summarization script
python financial_summarization.py

```

## 📖 Usage Guide

### Running the Pipeline

To execute the standard summarization and evaluation workflow:

```python
# In the notebook, run:
run_complete_pipeline(train_bart=False)

```

*Note: Set `train_bart=True` only if you wish to perform a full fine-tuning session, which requires significant GPU memory and time.*

### Integrating Custom Data

To use your own SEC or financial data, replace the `load_sample_data()` function. Ensure your data is provided as two lists:

* **Documents**: The full text to be summarized.
* **References**: The "gold standard" ground-truth summaries for comparison.
* **Recommendation**: Use 100+ documents for meaningful results.

## 📂 Project Structure

* **/results**: Stores generated PNG visualizations and performance CSV/TXT files.
* **/models**: Contains saved BART model checkpoints after fine-tuning.
* **Code File.ipynb**: The main entry point containing all logic and processing steps.
