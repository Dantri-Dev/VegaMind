from fastapi import FastAPI, HTTPException
import argparse
import os
import yaml
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from src.data_processing import DataProcessor
from src.embedding import EmbeddingHandler
from src.qdrant_client import QdrantHandler
from src.agent import VegaMindAgent
from src.tools.tool_generate_filters import ToolGenerateFilters
from src.tools.tool_generate_filters_sirius import ToolGenerateFiltersSirius
import numpy as np
import time
import logging

# Inizializzazione dell'app FastAPI
app = FastAPI()

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# Modello Pydantic per la richiesta della query
class QueryRequest(BaseModel):
    query: str

# Modello Pydantic per la risposta
class QueryResponse(BaseModel):
    result: str

# Funzione per elaborare una singola query
def process_single_query(query: str) -> str:
    # Carica la configurazione
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Inizializza i gestori
    embedding_handler = EmbeddingHandler()
    qdrant_handler = QdrantHandler()

    # Inizializza i tool che VegaMindAgent può usare
    tool_1 = ToolGenerateFilters()
    tool_2 = ToolGenerateFiltersSirius()

    # Inizializza l'agent
    agent = VegaMindAgent(
        tools={"generate_filters": tool_1, "generate_filters_sirius": tool_2},
        qdrant_handler=qdrant_handler
    )

    # Elabora la query e ottieni il risultato
    result = agent.process_query(0, query)

    # Restituisci solo la parte 'result' del dizionario
    return result['result']

# Endpoint per elaborare una singola query
@app.post("/process_query/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Endpoint per elaborare una singola query.
    """
    try:
        result = process_single_query(request.query)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione della query: {str(e)}")

# Endpoint per configurare il database (set up DB)
@app.post("/setup_db/")
async def setup_database():
    """
    Endpoint per configurare il database Qdrant.
    """
    try:
        # Carica la configurazione
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Elaborazione dei dati e configurazione del database
        print("Elaborazione dei documenti...")
        data_processor = DataProcessor()
        all_chunks, metadata = data_processor.process_all_documents()

        print("Generazione degli embeddings...")
        embedding_handler = EmbeddingHandler()
        embeddings = embedding_handler.generate_embeddings(all_chunks)

        # Verifica la struttura dell'embedding
        if isinstance(embeddings[0], np.ndarray):
            embedding_dim = embeddings[0].shape[0]
        elif isinstance(embeddings[0], list):
            embedding_dim = len(embeddings[0])
        else:
            raise ValueError("L'embedding non ha la struttura prevista")

        print("Configurazione del database Qdrant...")
        qdrant_handler = QdrantHandler()
        qdrant_handler.setup_collection(embedding_dim)

        print("Caricamento dei documenti in Qdrant...")
        payload = [{"text": chunk, **meta} for chunk, meta in zip(all_chunks, metadata)]
        qdrant_handler.upload_documents(embeddings, payload)

        print("Database configurato con successo!")
        return {"message": "Database configurato con successo!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la configurazione del database: {str(e)}")

# Se vuoi elaborare una query da file CSV, puoi aggiungere un altro endpoint
@app.post("/process_csv/")
async def process_csv():
    """
    Endpoint per elaborare domande da un file CSV.
    """
    try:
        # Carica la configurazione
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Inizializza i gestori
        embedding_handler = EmbeddingHandler()
        qdrant_handler = QdrantHandler()

        # Inizializza i tool che VegaMindAgent può usare
        tool_1 = ToolGenerateFilters()
        tool_2 = ToolGenerateFiltersSirius()

        # Inizializza l'agent
        agent = VegaMindAgent(
            tools={"generate_filters": tool_1, "generate_filters_sirius": tool_2},
            qdrant_handler=qdrant_handler
        )

        # Carica il file CSV con le domande
        questions_df = pd.read_csv(config["paths"]["questions"])

        # Rimuove spazi da tutti i nomi delle colonne
        questions_df.columns = questions_df.columns.str.strip()

        # Aggiungi una colonna 'row_id' se non esiste
        if 'row_id' not in questions_df.columns:
            questions_df['row_id'] = questions_df.index + 1

        # Elabora le domande
        results = []
        total_queries = len(questions_df)

        output_path = config["paths"]["output"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Controlliamo se il file esiste già
        file_exists = os.path.isfile(output_path)

        for index, row in questions_df.iterrows():
            query = row["domanda"]
            row_id = row["row_id"]

            result = agent.process_query(row_id, query)
            results.append(result)

            result_df = pd.DataFrame([result])  # Creiamo un DataFrame con una sola riga
            result_df.to_csv(output_path, mode="a", header=not file_exists, index=False)
            file_exists = True

            # Attendi 5 secondi prima di elaborare la prossima domanda
            time.sleep(5)

        return {"message": "Elaborazione completata! Risultati salvati."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione del CSV: {str(e)}")
