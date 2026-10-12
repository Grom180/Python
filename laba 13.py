import requests

MODEL = 'qwen2.5:1.5b'
def ask(prompt, system='', temperature=0.7):
    r = requests.post('http://localhost:11434/api/generate', json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'options': {'temperature': temperature}})
    return r.json()['response']
if __name__ == '__main__':
    print(ask('Объясни, что такое нейросеть, в двух предложениях.'))
# -
if __name__ == '__main__':
    print(ask('Ты ассистент колледжа. Отвечай кратко, вежливо, только по-русски.'))
    print(ask('Отвечай только ДА или НЕТ'))
# -
p = '''Определи категорию обращения одним словом.
Фраза: Где расписание на завтра? Категория: расписание
Фраза: Столовая сегодня работает? Категория: столовая
Фраза: Мне нужна справка для военкомата. Категория: документы
Фраза: Во сколько закрывается буфет? Категория:'''
print(ask(p, temperature=0))
# - обучение на ответ
import json
r = requests.post('http://localhost:11434/api/generate', json={
    'model': MODEL, 'stream': False, 'format': 'json',
    'prompt': 'Верни JSON с полями intent (одно слово) и answer (одно предложение) для запроса:  Какое сегодня расписание у колледжа НРТК?'})
print(json.loads(r.json()['response']))
