import torch
from sentence_transformers import SentenceTransformer
from src.config_loader import ConfigLoader
import logging

class EmbeddingHandler:
    def __init__(self, config_path="config/config.yaml"):
        print("\n[INIT] Caricamento della configurazione degli embedding...\n")
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        
        model_name = self.config["embedding"]["model"]
        self.chunk_size = self.config["embedding"]["chunk_size"]
        self.chunk_overlap = self.config["embedding"]["chunk_overlap"]
        self.add_instruction = self.config["embedding"].get("add_instruction", False)
        
        print(f"[INFO] Modello di embedding selezionato: {model_name}\n")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        print(f"[OK] Modello di embedding `{model_name}` caricato su `{self.device}`!\n")
        
        # Configura il logger
        logging.basicConfig(level=logging.INFO)  # Imposta il livello di log
        self.logger = logging.getLogger(__name__)

    def generate_query_embedding(self, text):
        """Genera embedding per la query, aggiungendo il prompt di retrieval se necessario."""
        self.logger.info("Inizio generazione embedding per il testo.")
        
        # Mostra solo i primi 100 caratteri
        self.logger.info(f"Testo per l'embedding: {text[:100]}...")

        # Genera l'embedding
        self.logger.info("Generazione dell'embedding in corso...")
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()

        self.logger.info("Generazione embedding completata.")
        return embedding
    
    def generate_embeddings(self, texts):
        """Genera gli embedding per una lista di testi."""
        self.logger.info("Inizio generazione embedding per i testi.")

        # Log dei primi 100 caratteri dei primi 3 testi
        self.logger.info(f"Testi per gli embedding: {', '.join([text[:100] for text in texts[:3]])}...")
        
        embeddings = []
        
        # Generazione degli embedding per ciascun testo
        self.logger.info("Generazione degli embedding in corso...")
        for text in texts:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            embeddings.append(embedding)
        
        self.logger.info("Generazione embedding completata.")
        return embeddings


    def generate_document_embedding(self, text):
        """Genera embedding per un documento."""
        formatted_text = f"Represent this sentence for retrieval: {text}"
        return self.model.encode(formatted_text, convert_to_numpy=True).tolist()
