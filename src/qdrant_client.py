import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.config_loader import ConfigLoader

class QdrantHandler:
    def __init__(self, config_path="config/config.yaml"):
        """Inizializza il client Qdrant con i parametri da config.yaml."""
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()

        try:
            self.client = qdrant_client.QdrantClient(
                host=self.config["qdrant"]["host"],
                port=self.config["qdrant"]["port"]
            )
            print(f"[OK] Connesso a Qdrant su {self.config['qdrant']['host']}:{self.config['qdrant']['port']}")
        except Exception as e:
            print(f"[ERRORE] Impossibile connettersi a Qdrant: {e}")

    def setup_collection(self, vector_size):
        """Configura o ricrea la collezione in Qdrant."""
        metric = self.config["qdrant"].get("metric", "cosine").upper()  # Default: COSINE
        metric_mapping = {
            "COSINE": Distance.COSINE,
            "DOT": Distance.DOT,
            "EUCLIDEAN": Distance.EUCLID
        }

        selected_metric = metric_mapping.get(metric, Distance.COSINE)

        try:
            self.client.recreate_collection(
                collection_name=self.config["qdrant"]["collection_name"],
                vectors_config=VectorParams(size=vector_size, distance=selected_metric)
            )
            print(f"[OK] Collezione `{self.config['qdrant']['collection_name']}` configurata con metrica `{metric}`")
        except Exception as e:
            print(f"[ERRORE] Errore nella creazione della collezione: {e}")

    def upload_documents(self, embeddings, payload):
        """Carica i documenti nella collezione Qdrant usando `upsert`."""
        try:
            points = [
                PointStruct(id=i, vector=vector, payload=payload[i])
                for i, vector in enumerate(embeddings)
            ]

            self.client.upsert(
                collection_name=self.config["qdrant"]["collection_name"],
                points=points
            )
            print(f"[OK] Caricati {len(points)} documenti nella collezione `{self.config['qdrant']['collection_name']}`")
        except Exception as e:
            print(f"[ERRORE] Problema durante l'upload dei documenti: {e}")

    def search(self, query_vector, k=5, qdrant_filter=None):
        """Esegue una ricerca vettoriale nella collezione Qdrant con filtri opzionali.
        
        Args:
            query_vector (List[float]): Il vettore di embedding della query.
            k (int): Numero di risultati da restituire (default: 5).
            qdrant_filter (Optional[models.Filter]): Filtro Qdrant per affinare la ricerca.
            
        Returns:
            List[models.ScoredPoint]: Lista dei risultati trovati.
        """
        try:
            results = self.client.search(
                collection_name=self.config["qdrant"]["collection_name"],
                query_vector=query_vector,
                limit=k,
                query_filter=qdrant_filter  # Aggiungi il filtro Qdrant
            )
            print(f"[OK] Ricerca completata: {len(results)} risultati trovati")
            return results
        except Exception as e:
            print(f"[ERRORE] Errore nella ricerca: {e}")
            return []
    
    def search_with_filters(self, qdrant_filter, k=5):
        """Esegue una ricerca in Qdrant basata solo sui filtri."""
        try:
            results = self.client.scroll(
                collection_name=self.config["qdrant"]["collection_name"],
                scroll_filter=qdrant_filter,
                limit=k,
                with_payload=True,
                with_vectors=False
            )
            return results[0]  # Restituisce la lista dei risultati
        except Exception as e:
            print(f"[ERRORE] Errore durante la ricerca con filtri: {e}")
            return []
