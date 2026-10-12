from sentence_transformers import SentenceTransformer
import numpy as np
import requests

MODEL = 'qwen2.5:1.5b'
def ask(prompt, system='', temperature=0.7):
    r = requests.post('http://localhost:11434/api/generate', json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'options': {'temperature': temperature}})
    return r.json()['response']

text = open('docs.txt', encoding='utf-8').read()
chunks = [text[i:i+400] for i in range(0, len(text), 350)]
print(len(chunks)); print(chunks[0])
# -
emb = SentenceTransformer('cointegrated/rubert-tiny2')
V = emb.encode(chunks, normalize_embeddings=True)
print(V.shape)
# -
def search(q, k=3):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    scores = V @ qv
    idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in idx]
for s, c in search('Сколько часов длится конкурс?'): print(round(s, 3), c[:100])
# -
def rag(q, k=2):
    ctx = '\n---\n'.join(c for _, c in search(q, k))
    p = f'Ответь на вопрос только по контексту. Если ответа в контексте нет, скажи: не знаю.\n\nКонтекст:\n{ctx}\n\nВопрос: {q}\nОтвет:'
    return ask(p, temperature=0)
print(rag('Сколько часов длится конкурс?'))
print(rag('Какая столица Австралии?'))  # ожидаем: не знаю
# читает инфу из файла и отвечает на вопросы