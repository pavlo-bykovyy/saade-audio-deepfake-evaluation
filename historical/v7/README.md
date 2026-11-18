# SAADE–AUA v7 — сценарно-адаптивне ТЕСТУВАННЯ, а не навчання

## Мета

Адаптація методики **Scenario-Adaptive Evaluation of Trustworthy Fine-Tuned Text Models Across Knowledge-Grounded Generation and Misinformation Detection** (DOI: https://doi.org/10.3390/make8060161) для аудіодетекторів. Жодні нейромережі не перенавчаються. Оцінюються попередньо навчені моделі із зафіксованими вагами.

## Вхідні дані на Jupyter-сервері

Використовується наявний `V6_LOCKED_MATCHED_SPLITS.csv` і фактично згенеровані WAV із:

`/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v6_2_MATCHED_TTS/`

Якщо підкорпус збережений під v6, код перевіряє також альтернативний каталог `SAADE_AUA_2026_v6_MATCHED_TTS`.

Оригінальний `/home/jovyan/work/audio_real_fake/AUA` **не змінюється**. Папка `/home/jovyan/work/Melnyk_dus` **не використовується**.

## Як почати

1. Скопіюйте **тільки `SAADE_AUA_v7_SCENARIO_EVALUATION_ALL_IN_ONE.ipynb`** у JupyterLab.
2. Якщо пакети відсутні, виконайте в окремій комірці `%pip install 'transformers>=4.40,<5' torch onnxruntime soundfile scipy scikit-learn statsmodels matplotlib pandas numpy` і перезапустіть kernel. Потрібен системний **FFmpeg** (наприклад, `conda install -c conda-forge ffmpeg`).
3. Перший прогін: залишіть `MAX_PAIRS = 12`, `EVAL_SPLITS = ('validation',)`, `RUN_INFERENCE = True`. `MODEL_ENABLED` визначає, які **готові** моделі запускати.
4. Запустіть **Restart Kernel and Run All Cells**. Результат: `/home/jovyan/work/audio_real_fake/results/SAADE_SCENARIO_EVALUATION_v7_SMOKE_12_FOR_REVIEW.zip`.
5. Для повного тестування змініть тільки `MAX_PAIRS = None`, перезапустіть від початку. Результат: `.../results/SAADE_SCENARIO_EVALUATION_v7_FULL_FOR_REVIEW.zip`.
6. Надішліть архів `FULL_FOR_REVIEW.zip` для перевірки і планування статті до журналу **AI** (MDPI).

### Моделі

- **AASIST ONNX** (~1,62 MB): https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST
- **AASIST-L ONNX** (~766 KB; споріднена, полегшена архітектура): https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L
- **Wav2Vec2** (~1,26 GB): https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector
- **Hemgg** — додаткова опція, за замовчуванням вимкнена через необхідність незалежно верифікувати орієнтацію класів: https://huggingface.co/Hemgg/Deepfake-audio-detection

Код використовує оригінальні checkpoint; випадкових ваг немає. Якщо Hugging Face знову видасть `TypeError: process() takes no keyword arguments`, реалізовано fallback-завантаження через `urllib` без Brotli. Якщо мережевий доступ до файлів недоступний, ноутбук повідомить про помилку — **не підмінить модель випадковими вагами**.

## Методика

- 21 сценарій: clean + 5 рівнів шуму (SNR) + 5 рівнів справжньої MP3-компресії + 5 рівнів синтетичної реверберації (RT60) + 5 lowpass частот.
- Усі сценарії застосовуються до кожної REAL/FAKE пари, з фіксованими параметрами.
- 16 kHz mono, до 4 с; **спільна корисна тривалість усередині пари**.
- Поріг встановлюється **лише за clean train** (інференс, без модифікації ваг), а результати оцінюються на `validation`.
- Метрики: Balanced Accuracy, ROC-AUC, EER, F1(fake), sensitivity, specificity, confusion matrix.
- **ASRI-Audio** (нове визначення, не ARI першоджерела): середнє Balanced Accuracy 4 категорій по 5 рівнях, *без clean*.
- Оцінювання надійності: pair-bootstrap, repeated-measures ANOVA (якщо достатньо незалежних пар), кластер-робастний факторний аналіз, MCDM (SRD/DnE/DnM/WSM/TOPSIS), зміна рейтингу між сценаріями.
- Графіки та відтворення реального й синтетичного аудіо — у ноутбуці.
- `hidden_test` **не використовується** в цьому випуску. Код зупиниться, якщо його додати в `EVAL_SPLITS`.

## Наукові обмеження

У наявній v6 validation є **лише один незалежний TTS-голос**. Pair-bootstrap не можна подавати як довірчі інтервали узагальнення на інші голоси або генератори. AASIST і AASIST-L — споріднені моделі; окремим сімейством є Wav2Vec2. Для повноцінної Q1-статті потрібні нові незалежні синтетичні голоси/генератори та зовнішній тест. Ознаки каналу запису REAL проти TTS також можуть впливати на виміряні результати.

Перший прогін на `MAX_PAIRS=12` є **технічним smoke-test**, не публікаційним результатом.

## Стан перевірки

Формат і синтаксис ноутбука перевірено. Проведено **наскрізний офлайн-тест на штучно зібраній файловій структурі з тимчасовими WAV та незмінюваними mock-предикторами**: всі 21 акустичний сценарій, балансування пар, калібрування, метрики, графіки й ZIP. **Актуальні checkpoint AASIST/Wav2Vec2 на вашому сервері в цьому тесті не запускалися**, а оригінальний AUA у локальному середовищі недоступний. Результати моделювання є перевіркою коду, не результатами дослідження.
