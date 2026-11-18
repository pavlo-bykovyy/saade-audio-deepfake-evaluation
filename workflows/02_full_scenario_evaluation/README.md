# SAADE–AUA v9 — Повний експеримент без пілотних обмежень

Відкрийте `SAADE_AUA_v9_COMPLETE_ALL_DATA_SCENARIO_EVALUATION.ipynb` в JupyterLab на сервері, де доступний `/home/jovyan/work/audio_real_fake/AUA`, і виконайте **Kernel → Restart Kernel and Run All Cells**.

За замовчуванням `RUN_PHASE=FULL`, `INCLUDE_ALL_AUA=True`, `INCLUDE_ALL_PUBLIC_MATCHED=True`, `ALL_NONOVERLAPPING_SEGMENTS=True`; жодних обмежень кількості пар чи первинних файлів. Усю роботу буде записано в `audio_real_fake/results/SAADE_AUA_2026_v9_COMPLETE_EVALUATION/FULL_EXPERIMENT`. Встановлено 3 готові моделі: AASIST, AASIST-L, Wav2Vec2 Stafford; усі 21 сценарій. НІЯКОГО НАВЧАННЯ.

Після завершення надсилайте `audio_real_fake/results/SAADE_AUA_v9_COMPLETE_ALL_DATA_FOR_REVIEW.zip`. Великий файл містить розгорнуті прогнози та методичні звіти без WAV, сирих транскрипцій або абсолютних шляхів.

Hidden не включено у FULL навмисно: він зберігає науково незалежний фінальний тест. Коли повний звіт перевірено: `RUN_PHASE=FREEZE`, вказати `FREEZE_CONFIRMATION=I_APPROVE_COMPLETE_PROTOCOL`, запустити. Потім `RUN_PHASE=HIDDEN_ONCE`, вказати `HIDDEN_CONFIRMATION=I_AUTHORIZE_ONE_FINAL_HIDDEN_TEST`, знову запустити. Hidden відкривається один раз і дає окремий ZIP.

**Обмеження:** За відомим складом набору matched-validation представлений лише одним TTS-голосом. AUA має зв'язок джерела з міткою; тому для нього наведено профілі для кожного джерела окремо, а основне міжмодельне порівняння — на matched-validation. Ці обмеження не прибираються збільшенням кількості сценаріїв.

**Середовище:** `numpy`, `pandas`, `scipy`, `soundfile`, `scikit-learn`, `matplotlib`, `torch`, `transformers (<5)`, `onnxruntime`, `huggingface_hub`, `statsmodels`, `safetensors`, CLI `ffmpeg/ffprobe`. У разі відсутності Python-залежностей установіть `INSTALL_MISSING=True` один раз. Перед запуском переконайтеся у достатньому просторі для SQLite-кешу та графіків. Повний прогін потребує значного ресурсу CPU/GPU. Зупинений прогін можна відновити, не перезаписуючи готові прогнози.
