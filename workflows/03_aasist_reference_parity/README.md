# SAADE–AUA — AASIST ONNX ↔ PyTorch parity audit, виправлення 404

Цей ноутбук **не змінює** повний експеримент v9 і **не відкриває** hidden test.

## Причина 404
У репозиторії Hugging Face `SpeechAntiSpoofingBenchmarks/AASIST` відсутні `_net.py` і `aasist.py`. Попередній аудит помилково намагався їх завантажити. У виправленому аудиті мережеву архітектуру PyTorch беруть з офіційного `clovaai/aasist/models/AASIST.py`, за потреби — з наявного файлу AASIST-L `_net.py`. Актуальні об'єкти конфігурацій AASIST і AASIST-L відтворено з офіційних `config/AASIST.conf` та `config/AASIST-L.conf`; `.pth` завантажуються з відповідних моделей Hugging Face. ONNX використовуються **виключно ті**, які вже працювали в v9, за зафіксованими SHA-256.

## Виконання
1. Завантажити `SAADE_AASIST_PARITY_AUDIT_FIXED_v2.ipynb` у JupyterLab.
2. Обрати **Kernel → Restart Kernel and Run All Cells**.
3. Підключення до мережі потрібне лише для одержання офіційного коду PyTorch і двох `.pth`, якщо їх ще немає у кеші аудиту.
4. Перевірити статус `PASS` або `CHECK` в останній комірці.

Каталог результатів:
`/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v9_COMPLETE_EVALUATION/AASIST_REFERENCE_PARITY_AUDIT/`

Результат містить:
- `00_REFERENCE_PROVENANCE.json` — походження і контрольні суми джерела/ваг;
- `01_ONNX_PYTORCH_PARITY_PER_RECORD.csv` — обидва логіти на 12 REAL–FAKE парах;
- `02_PARITY_SUMMARY.csv` — збіг ONNX/PyTorch (головний висновок);
- `03_PADDING_EFFECT.csv` — вплив zero-padding порівняно з repeat-padding;
- `03_PADDING_EFFECT_PER_RECORD.csv` — ці зміни на кожному записі;
- `04_EXISTING_V9_SCORE_CHECK.csv` — порівняння з реально збереженими балами v9;
- `05_AUDIT_DECISION.json` — зведений статус.

**Статус `CHECK` не означає, що весь v9 треба перераховувати.** Він означає, що певний елемент парності або відповідності старих оцінок треба з'ясувати перед фіксацією протоколу.

Джерела архітектури: https://github.com/clovaai/aasist і https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L/tree/main.
