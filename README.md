# AnalyTIQS-Agents

## Project Overview

AnalyTIQS-Agents brings together two conversational, AI-powered agents for fast, code-free data analytics and visualization on any CSV file in your directory.  
- **LCEL Data Analytics Agent:** A fully conversational tool-based agent that guides users—step by step—through data import, cleaning, model building (classification or regression), and results explanation.  
- **Pandas DataFrame Visualization Agent:** Instantly lists your local CSV files for selection, then enables users to chat naturally for on-demand analysis, quick stats, and auto-generated charts.

***

## Key Features & Highlights

- **Conversational, Context-Aware AI:**  
  Both agents use a multi-turn, interactive chat interface for all data science tasks, delivering explanations and actionable insights in plain language.

- **Automatic CSV File Discovery:**  
  On startup, each agent auto-detects and lists all CSV files in the current working directory—making your data instantly accessible.

- **Tool-Oriented Analytics (LCEL Agent):**  
  - Modular tools handle individual stages like data cleaning, feature engineering, model training, and metric reporting.
  - Chained together by LangChain Expression Language (LCEL), so workflows are composable and extensible.
  - Explains each step and lets you ask questions, rerun stages, or iterate with new settings—all in conversation.

- **Conversational Visualization (Pandas Agent):**  
  - Select any CSV, then ask free-form questions about your data.
  - Automatically generates descriptive stats or visualizations (bar plots, trends, distributions) by leveraging python, pandas, and matplotlib.
  - Keeps the whole flow conversational, with full Llama-3/Groq reasoning power on both chat and code generation.

***

## Workflow

1. **CSV File Selection:**  
   Both agents detect all `.csv` files detected in the working directory for seamless, mention-to-analyze working.

2. **LCEL Data Analytics Agent:**
   - Guides the user through choosing their ML task and selecting dataset columns.
   - Automatically performs data preparation, builds the chosen model, and presents evaluation results (classification/report generation or regression/metrics).
   - Follows up with context-rich summaries, key feature explanations, and further user-driven Q&A for deeper insight.

3. **Pandas Visualization Agent:**
   - Loads the selected file and enters a conversational loop.
   - Users ask about distributions, trends, group statistics, or request specific charts—agent produces code and visuals instantly.
   - Supports iterative, back-and-forth exploration with context memory.

***

## Results

- **No-code intelligent analytics, modeling, and visualization for any local CSV.**
- **Stepwise, plain-language conversations for guided exploration or charting.**
- **Modular, extensible design—easy to upgrade with new tools, pipelines, or chart types.**

***

*AnalyTIQS-Agents demonstrates how state-of-the-art LLMs and modern agent architectures can make analytics and visualization fully interactive and user-centric—for business, research, or education.*
