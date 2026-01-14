# -*- coding: utf-8 -*-
"""
Assistente de Voz Multi-Idiomas com Whisper e ChatGPT
Baseado no artigo da DIO: https://web.dio.me/articles/conversando-por-voz-com-o-chatgpt-utilizando-whisper-openai-e-python
"""

import os
import whisper
import openai
from gtts import gTTS
from IPython.display import Audio, display, Javascript
from google.colab import output
from base64 import b64decode

# --- CONFIGURAÇÕES ---
language = 'pt'
# Segurança: A chave é lida das variáveis de ambiente do sistema
openai.api_key = os.environ.get('OPENAI_API_KEY') 

# --- 1. GRAVAÇÃO DE ÁUDIO (JavaScript Interoperability) ---
# Fonte da lógica de gravação: https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be
RECORD = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec=5):
    display(Javascript(RECORD))
    js_result = output.eval_js('record(%s)' % (sec * 1000))
    audio = b64decode(js_result.split(',')[1])
    file_name = 'request_audio.wav'
    with open(file_name, 'wb') as f:
        f.write(audio)
    return f'/content/{file_name}'

# --- FLUXO DE EXECUÇÃO ---

# Passo 1: Captura de Voz
print('Ouvindo...')
record_file = record()

# Passo 2: Transcrição com Whisper (OpenAI)
model = whisper.load_model("small")
result = model.transcribe(record_file, fp16=False, language=language)
transcription = result["text"]
print(f"Você disse: {transcription}")

# Passo 3: Processamento com ChatGPT (GPT-4) e Contexto de Programação
contexto = "Você é um especialista em programação. Responda de forma técnica e objetiva."

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": contexto},
        {"role": "user", "content": transcription}
    ]
)

chatgpt_response = response.choices[0].message.content
print(f"ChatGPT: {chatgpt_response}")

# Passo 4: Conversão de Texto para Voz (gTTS)
gtts_object = gTTS(text=chatgpt_response, lang=language, slow=False)
response_audio = "/content/response_audio.wav"
gtts_object.save(response_audio)
display(Audio(response_audio, autoplay=True))
