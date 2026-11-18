# SAADE–AUA v6.2: виправлення Piper і TTS-голосів

**Файл:** `SAADE_AUA_ALL_IN_ONE_v6_2_PIPER_FIXED.ipynb`. Один самодостатній ноутбук.

## Що виправлено

1. Офіційний файл `uk_UA-lada-x_low.onnx.json` є правильним навіть без `phoneme_type`: перевірка використовує `audio.sample_rate`, `phoneme_id_map`, `num_speakers` та `espeak` або `phoneme_type`.
2. Усі п'ять моделей Piper присутні в актуальному `rhasspy/piper-voices`: `lada`, `mykyta`, `oleksa`, `tetiana`, `ukrainian_tts`.
3. `ukrainian_tts` має три голоси: Lada=0, Mykyta=1, Tetiana=2. Для цієї моделі код задає `--speaker 1` (Mykyta) і зберігає акустичну особу окремо від назви моделі. Тому два варіанти Mykyta не потраплять в різні вибірки.
4. Поточний дизайн: train = Oleksa, Mykyta (дві модельні конфігурації Mykyta тільки в train), validation = Tetiana, hidden = Lada. Adaptation відокремлюється від train на рівні незалежних голосових компонентів.
5. Повторно використовується вже завантажена модель `lada`, якщо вона є в результатах попередньої v6.
6. Результати v6.2 зберігаються **окремо** в `/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v6_2_MATCHED_TTS`. Попередні експерименти не переписуються. AUA та папка Melnyk_dus не змінюються.

## Запуск

Завантажте `.ipynb`, у JupyterLab натисніть `Kernel → Restart Kernel and Run All Cells`. Перший етап використовує AUA `/home/jovyan/work/audio_real_fake/AUA`. За наявності Інтернету завантажуються 5 справжніх ONNX моделей. Для великих high-quality моделей потрібне стабільне з'єднання та місце на диску.

Після генерації перевірте `V6_TTS_MODEL_PROVENANCE.csv`, `V6_TTS_RECEIPTS.csv`, `V5_01_ADAPTATION_PROVENANCE.json` і звітний ZIP.

**Наукове обмеження:** це контрольована Text→TTS вибірка, проте з однією незалежною синтетичною особою у validation та hidden. Строга багатоголосова статистика й одноразовий hidden test мають залишатися заблокованими до розширення набору.

**Статус перевірки:** виконана валідація структури/синтаксису ноутбука й тестування основних допоміжних функцій на локальних фікстурах. Фактичне мережеве завантаження ONNX і 600 генерацій на вашому сервері ще не перевірене.
