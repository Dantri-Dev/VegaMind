import chainlit as cl
import requests

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_resume
async def on_chat_resume(thread):
    pass
    
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Quali piatti contengono i Ravioli al Vaporeon?",
            message="Quali piatti contengono i Ravioli al Vaporeon?",
            ),

        cl.Starter(
            label="Quali sono i piatti della galassia che contengono Latte+?",
            message="Quali sono i piatti della galassia che contengono Latte+?",
            ),
        cl.Starter(
            label="Quali piatti dovrei scegliere per un banchetto a tema magico che includa le celebri Cioccorane?",
            message="Quali piatti dovrei scegliere per un banchetto a tema magico che includa le celebri Cioccorane?",
            ),
        cl.Starter(
            label="Quali piatti includono Essenza di Tachioni e Carne di Mucca, ma non utilizzano Muffa Lunare?",
            message="Quali piatti includono Essenza di Tachioni e Carne di Mucca, ma non utilizzano Muffa Lunare?",
            )
        ]

@cl.step(type="tool")
async def loading_tool(message_content):
    """
    Questo tool simula un'elaborazione in corso con un effetto di caricamento
    e invia una richiesta a un'API FastAPI.
    """
    
    try:
        # Prepara i dati per la richiesta 
        data = {
            "query": message_content  # Usa il contenuto del messaggio dell'utente
        }
        
        # Invia la richiesta HTTP al server FastAPI
        url = "http://127.0.0.1:8000/process_query"
        response = requests.post(url, json=data)
        # print(response.json())
        # Verifica se la risposta è valida
        if response.status_code == 200:
            risposta = response.json().get("result", "Non c'è risposta.")
        else:
            risposta = f"Errore nella richiesta: {response.status_code}"
            
        return risposta
        
    except Exception as e:
        return f"Si è verificato un errore durante l'elaborazione: {str(e)}"

@cl.on_message
async def main(message: cl.Message):
    """
    Questa funzione è chiamata quando un utente invia un messaggio.
    """
    
    # Chiama il tool di caricamento passando il contenuto del messaggio
    tool_result = await loading_tool(message.content)
    
    # Invia il risultato finale ottenuto dall'API
    await cl.Message(content=tool_result).send()