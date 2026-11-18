# SAADE–AUA v8.0 FINAL — Scenario-Adaptive Evaluation

**Остаточний код методики тестування, не модель навчання.** Один самодостатній файл `SAADE_AUA_v8_FINAL_SCENARIO_EVALUATION.ipynb` для JupyterLab. Базовий метод: Lipianina-Honcharenko et al., *Machine Learning and Knowledge Extraction* (2026), DOI: https://doi.org/10.3390/make8060161. Для аудіо використано власну адаптацію індексу ARI — `ASRI-Audio`, а не буквальне перенесення числових оцінок із задачі LLM.

## 1. До запуску

- Уже виконано v6.2 із реально створеними REAL–TTS WAV, `V6_LOCKED_MATCHED_SPLITS.csv`, `V6_TTS_RECEIPTS.csv`.
- Наявні `/home/jovyan/work/audio_real_fake/AUA` (оригінальний набір; **лише читання**) і `/home/jovyan/work/audio_real_fake/results`.
- Встановлені Python 3.10+, `numpy`, `pandas`, `scipy`, `soundfile`, `scikit-learn`, `matplotlib`, `torch`, `transformers` (4.x), `onnxruntime`, `huggingface_hub`, `statsmodels`, `safetensors`, а також системна програма **ffmpeg**.
- Доступ до pretrained моделей Hugging Face під час першого завантаження. Wav2Vec2 Stafford має великий файл ваг (близько 1,26 GB); його завантаження та інференс потребують належної RAM/VRAM.
- Якщо сервер має проблему `httpx2/Brotli`, адаптер Wav2Vec2 передбачає резервне завантаження `config.json`, `preprocessor_config.json`, `model.safetensors` через HTTPS без стиснення `Brotli`.

## 2. Перший (основний) запуск

1. Відкрийте `.ipynb` у JupyterLab.
2. У конфігураційній комірці на початку залиште:

   ```python
   RUN_PHASE = 'VALIDATION'
   MODEL_ENABLED = {'AASIST_ONNX': True, 'AASIST_L_ONNX': True, 'WAV2VEC2_STAFFORD': True}
   CALIBRATION_PAIRS = 48
   FREEZE_CONFIRMATION = 'NO'
   HIDDEN_CONFIRMATION = 'NO'
   ```
3. Виберіть **Kernel → Restart Kernel and Run All Cells**.
4. Наприкінці ноутбук сформує **один ZIP-файл** у `/home/jovyan/work/audio_real_fake/results/`:

   `SAADE_SCENARIO_EVALUATION_v8_FULL_VALIDATION_FOR_REVIEW.zip`

5. Надішліть архів для перевірки результатів. Він не містить WAV, checkpoint або первинних транскрипцій.

**Очікуваний обсяг відповідно до попереднього маніфесту v6.2:** 94 validation-пари × 2 класи × 21 сценарій × 3 моделі = **11 844 прогнозів**. Програма перевіряє фактичну кількість автоматично; 94 — не жорстко закодована константа.

## 3. Додатковий швидкий технічний запуск

Установіть `RUN_PHASE='SMOKE'` і `SMOKE_PAIRS=12`. Результати зберігаються в окремому каталозі, окремому від FULL. Не подавайте SMOKE як підсумковий дослідницький результат.

## 4. Незалежність результатів

- Зберігаються SHA-256 джерельних WAV і pretrained checkpoint; за невідповідності запуск блокується.
- Усі моделі обробляють ті самі 16-kHz/4-s аудіовікна, з однаковою корисною довжиною REAL/FAKE усередині пари.
- Чотири групи спотворень по п'ять рівнів і один clean; MP3 проходить реальний encode/decode через ffmpeg. Однаковий шумовий сигнал генерується для обох класів однієї пари на відповідному SNR.
- `ASRI-Audio` = середнє Balanced Accuracy по чотирьох типах спотворень; clean винесено окремо. Для кожної категорії усереднюються п'ять рівнів.
- CI за `pair_id` — **умовні**, а не статистичне підтвердження стійкості до довільних генераторів. Один Piper-голос `tetiana` у validation залишається суттєвим обмеженням.
- Аномальний ROC-AUC (особливо Wav2Vec2 у шумових сценаріях) автоматично **не інвертується** — потрібна перевірка передобробки та орієнтації класів.
- MCDM наводить п'ять індикаторів, як методологічний орієнтир першоджерела, але у підсумковому консенсусі не подвоює тотожні за цією нормалізацією DnM і WSM.

## 5. Hidden test: тільки після наукового перегляду FULL

**Не відкривайте hidden відразу.** Код містить явні захисні підтвердження.

- Після перегляду FULL змініть `RUN_PHASE='FREEZE'`, `FREEZE_CONFIRMATION='I_APPROVE_THE_VALIDATION_PROTOCOL'`, `HIDDEN_CONFIRMATION='NO'` і виконайте ноутбук. Створиться незмінний `FROZEN_PROTOCOL_v8.json`.
- На окремому запуску задайте `RUN_PHASE='HIDDEN_ONCE'`, `FREEZE_CONFIRMATION='NO'`, `HIDDEN_CONFIRMATION='I_AUTHORIZE_ONE_FINAL_HIDDEN_TEST'`. Код перевірить hash протоколу й виконає hidden один раз. При збоях одноразовий маркер **не видаляйте**.
- Повторна спроба автоматично блокується, навіть якщо попереднє виконання було перерване. Завершений hidden звіт: `SAADE_SCENARIO_EVALUATION_v8_HIDDEN_ONCE_FOR_REVIEW.zip`.

## 6. Чого цей ноутбук НЕ робить

- Не перенавчає, не fine-tune і не змінює ваги аудіодетекторів.
- Не переписує оригінальний AUA, не торкається `/home/jovyan/work/Melnyk_dus`.
- Не створює додаткові TTS-рушії: використовує наявний matched-TTS корпус v6.2.
- Не доводить міжгенераторну узагальнюваність на одному TTS-рушії Piper і одному synthetic validation-голосі.

## 7. Стан перевірки цього артефакту

Перевірено структуру `.ipynb`, синтаксис усіх 13 кодових комірок і наскрізний процес із фіксованими **тестовими** предикторами на синтетичних контрольних WAV: 21/21 сценаріїв, контроль SHA, оцінювання, графіки, ZIP, `VALIDATION→FREEZE→HIDDEN_ONCE` і блокування повторного hidden. **Фактичний інференс великих pretrained checkpoint на серверному AUA ще не виконувався цією версією**. Виконані контрольні тести не є науковими результатами.
