# AgenticAI-Amazon-Financial-Insights-HFL

## Overview

This project is an interactive, human-in-the-loop AI system for analyzing Amazon trading data. It leverages OpenAI's GPT-4o-mini, LangGraph for workflow orchestration, and vector embeddings for intelligent memory and semantic search. The system allows users to iteratively analyze, enhance, and visualize their data with natural language feedback.

---

## Features

- **Human-in-the-Loop Workflow**: Every analysis step is guided by user feedback.
- **Natural Language Interface**: Ask questions, request enhancements, or visualizations in plain English.
- **Intelligent Routing**: LLM decides the next action based on your feedback.
- **Semantic Memory**: Uses vector embeddings to recall and reuse similar past analyses.
- **Professional Visualizations**: Generates line and bar charts for your data.
- **Persistent State**: Saves all sessions, analyses, and embeddings for future use.

---

## Quick Start

### 1. **Install Requirements**

```bash
pip install openai pandas matplotlib python-dotenv pydantic langgraph numpy
```

### 2. **Set Up API Keys**

Create a `.env` file in the project directory with your OpenAI API key:

```
OPENAI_API_KEY=your-openai-key-here
```

### 3. **Prepare Your Data**

- Place your Amazon trading CSV file in the project directory.
- The system will automatically use the first 200 columns for analysis.

### 4. **Run the Application**

```bash
python AI_Agent.py
```

### 5. **Interact**

- Enter the path to your CSV file when prompted.
- Ask your analysis question (e.g., "Find the top 10 products by net profit margin").
- Provide feedback or request charts in natural language.
- The system will iterate based on your input until you end the session.

---

## Project Structure

```
project/
├── AI_Agent.py                # Main application code
├── README.md                  # This file
├── .env                       # Your API keys (not committed)
├── amazon.csv                 # Example data file
├── charts/                    # Generated chart images

```

---

## How It Works

1. **Data Loading**: Loads your CSV, auto-detects columns, and prepares for analysis.
2. **Initial Analysis**: Uses LLM to analyze your query and generate insights.
3. **Human Feedback**: Pauses for your feedback at every step.
4. **LLM Routing**: Decides whether to enhance, visualize, analyze new data, or end based on your input.
5. **Visualization**: Generates professional charts as requested.
6. **Memory**: Remembers similar queries and reuses results for efficiency.

---

## Documentation

- See `CODE_DOCUMENTATION.md` for a detailed breakdown of every class and function.
- See `LANGGRAPH_WORKFLOW.md` for a deep dive into the LangGraph workflow and state management.

---

## Example Usage

```
🤔 What would you like to analyze? Find the top 10 products by net profit margin
AI: [Analysis and insights]
Your feedback: Show me a chart of these results
AI: [Bar chart generated and saved]
Your feedback: I'm done
AI: [Session ends, summary displayed]
```

---

## Requirements

- Python 3.8+
- OpenAI API key

---

## Notes

- The system is designed for flexibility and extensibility. You can add new analysis or visualization types by extending the relevant classes.
- All data and results are stored locally for privacy and reproducibility.

---
