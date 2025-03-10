## VegaMind: An AI Assistant for Intergalactic Explorers
This FastAPI application provides an API for processing natural language queries using the **VegaMind** agent, which leverages vector search through **Qdrant**.
It enables advanced **search and filtering** capabilities to retrieve dishes based on specific ingredients, techniques, and contextual information.  
<br>
<p align="center">
  This project was developed for <strong>Hackapizza 2025 🍕 - Community Edition</strong>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9f9faf09-72ae-4747-9f5f-ff92e52b8e34" alt="Screenshot" width="480">
</p>

## Video Demonstration

[VegaMind-VideoDemo.webm](https://github.com/user-attachments/assets/05eb93e3-0249-4f61-b709-96af9f5cc9c1)


## Overview

This application facilitates natural language query processing by:
1. Setting up a vector database (Qdrant) with document embeddings
2. Processing user queries against the vector database
3. Generating appropriate filters for search queries
4. Supporting both single query processing and batch processing from CSV files

## Requirements

- Python 3.8+
- FastAPI
- Qdrant
- Pandas
- PyYAML
- Pydantic
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/Dantri-Dev/VegaMind
cd VegaMind

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```bash
VegaMind-Project/
├── config/                    # Configuration folder
│   └── config.yaml            # Configuration file
├── src/                       # Main folder containing the core logic of the project
│   ├── agent.py               # Implementation of the VegaMindAgent
│   ├── data_processing.py     # Data processing utilities
│   ├── embedding.py           # Embedding generation handler
│   ├── qdrant_client.py       # Qdrant client for connecting to the database
│   └── tools/                 # Folder for tool implementations
│       ├── tool_generate_filters.py   # Tool to generate filters
│       └── tool_generate_filters_sirius.py   # Tool to generate Sirius filters
├── main.py                    # Main FastAPI application
├── VegaMindChat/              # VegaMindChat Module for chat functionality
│   └── app/                   # Chat application folder
│       └── app.py             # Chat logic implementation
│   └── README.md              # Instructions for configuring VegaMindChat
└── requirements.txt           # Project dependencies
```

## API Endpoints

Process Single Query
```bash
POST /process_query/
```
Setup Database
```bash
POST /setup_db/
```

Initialize the Qdrant database with document embeddings.
This endpoint:
```bash
1. Processes all documents
2. Generates embeddings
3. Sets up the Qdrant collection
4. Uploads documents with their embeddings
```

Process CSV
```bash
POST /process_csv/
```

## Process multiple queries from a CSV file.
The endpoint:
1. Loads questions from the CSV file specified in the config
2. Processes each query
3. Saves results to the output file
4. Waits 5 seconds between processing each question

## VegaMindChat Setup

To correctly configure the VegaMindChat module, please follow the installation guide provided in the official [Chainlit Datalayer repository](https://github.com/Chainlit/chainlit-datalayer).
