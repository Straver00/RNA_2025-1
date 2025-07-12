from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import base64
import numpy as np

# Instrucción del sistema para el modelo
with open("system_instruction.txt", "r") as file:
    system_instruction = file.read().strip()


# Función para inicializar el cliente de la API Gemini
def init_client(api_key_client):
    """
    Inicializar el cliente API Gemini con la clave de API disponible.

    :return: Cliente de la API Gemini o None si no se pudo inicializar
    """
    if not api_key_client:
        print("La clave de API no está configurada. Por favor, verifica tu variable de entorno.")
        return None
    
    # Inicializar el cliente GenAI
    try:
        client = genai.Client(api_key=api_key_client)
        print("Cliente GenAI inicializado correctamente.")
        return client
    except ImportError:
        print("Por favor, instala el paquete 'google-genai' para utilizar esta funcionalidad.")
        return None
    except Exception as e:
        print(f"Se produjo un error al inicializar el cliente GenAI: {e}")
        return None
    except:
        print("Se produjo un error desconocido. Verifique su variable de entorno con la clave de API.")
        return None


# Función para generar una historia con el modelo Gemini con lenguaje natural
def generate_story_from_natural_language(client, user_prompt, creativity = 0.5):
    """
    Genera una historia utilizando el modelo Gemini.

    :param client: Cliente de la API Gemini
    :param user_prompt: Prompt del usuario para generar la historia
    :param creativity: Nivel de creatividad del modelo (0.0 a 1.0)
    :return: Tupla con la historia generada y el prompt de imagen
    """
    
    if not client:
        print("El cliente GenAI no está inicializado.")
        return None

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            seed=42,
            temperature = creativity,
            system_instruction=system_instruction,
            # Desactiva el pensamiento avanzado (no re requiere y gasta muchos tokens)
            thinking_config=types.ThinkingConfig(thinking_budget=0),  
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        ),
    )

    history = response.text.split("PROMPT_IMAGEN: ")[0]
    prompt_image = response.text.split("PROMPT_IMAGEN: ")[1]
    
    return history, prompt_image


# Función para generar una imagen a partir de un prompt
def generate_image_from_prompt(client, prompt_image, creativity = 0.8, verbose = False):
    """
    Genera una imagen utilizando el modelo Gemini a partir de un prompt.

    :param client: Cliente de la API Gemini
    :param prompt: Prompt para generar la imagen
    :param creativity: Nivel de creatividad del modelo (0.0 a 1.0)
    :return: Imagen generada como objeto np.array
    """
    
    if not client:
        print("El cliente GenAI no está inicializado.")
        return None

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt_image,
        config=types.GenerateContentConfig(
            response_modalities = ['TEXT', 'IMAGE'],
            seed=42,
            temperature = creativity,
        ),
    )

    image = None
    for part in response.candidates[0].content.parts:
        if part.text is not None and verbose:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            # image.save('gemini-native-image.png')
            # image.show()

    if image is None and verbose:
        print("No se pudo generar la imagen. Verifique el prompt o la configuración del modelo.")
        return None

    # Convertimos la imagen a un objeto np.array
    image_np = np.array(image)
    
    return image_np