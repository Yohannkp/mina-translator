"""Test API loading"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print('Testing model load...')
from api.main import load_model, translate_text

print('Loading model...')
load_model()

print()
print('Testing translation...')
text = 'Bonjour, comment allez-vous?'
translation, time_ms = translate_text(text)
print(f'FR: {text}')
print(f'MINA: {translation}')
print(f'Temps: {time_ms:.1f} ms')
print('SUCCESS!')