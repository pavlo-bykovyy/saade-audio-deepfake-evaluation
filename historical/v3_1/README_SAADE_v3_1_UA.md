# SAADE–AUA v3.1 — виправлення шляхів і DRSSU

**Чинні шляхи на сервері:**

- Дані (лише читання): `/home/jovyan/work/audio_real_fake/AUA`
- Результати: `/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v3_1_metadata_fixed`
- Папка Мельника `/home/jovyan/work/Melnyk_dus/` **не використовується для запису й не змінюється**. Наявні аудити `AUA_DEEP_AUDIT_SUMMARY.txt`, `FINAL_KPI_ACCEPTANCE.csv`, `AUA_FULL_AUDIT.xlsx` можна переглядати окремо як довідкові файли, але новий конвеєр не копіює та не змінює їх.

**Виправлення порівняно з v3:** для DRSSU `source_record_id = drssu:<selection_rank>` зіставляється з `selection_rank` у вихідному `metadata.csv`, перевіряється первинний SHA-256, повторні SHA виключаються з експерименту. Нові TTS без підтвердженого `voice_id` використовуються лише в TRAIN за умови підтвердженої назви моделі в реєстрі. Для FLEURS/DRSSU технічні `S1108`/`S1109` не вважаються реальними мовцями.

**Числа 112 219 файлів / 20,403 GiB — повна мультимодальна структура AUA**, а не число незалежних аудіоприкладів. Основний audio pipeline навмисно читає тільки `01_audio/{own_processed,external_selected,synthetic}`. `05_splits` містить копії/похідні дані; `00_sources_and_licenses` — первинні джерела та метадані; вони не додаються вдруге до навчання.

**Для запуску:** завантажте/розпакуйте всі файли ZIP в одну папку, перезапустіть Jupyter kernel; спочатку виконайте `SAADE_AUA_v3_1_Preflight_Data_Audit.ipynb`, а потім `SAADE_AUA_v3_1_Full_Pipeline.ipynb`. Переглядач `SAADE_AUA_v3_1_Results_Viewer.ipynb` не вимагає нового навчання. `EXPERIMENT_MODE='pilot'`, `RUN_REPEATS=False`, `RUN_FINAL_HIDDEN_TEST=False` — початкові безпечні параметри. Ніякої експертної оцінки немає.

**Обмеження:** виправлення підключення даних не усуває змішування `source × label`. За відсутності джерела з обома класами це діагностичний пілот; для статті знадобляться незалежні порівнювані real/fake, 3–5 seeds та зафіксований протокол перед hidden test.

---

# SAADE–AUA v3 — виправлене підключення метаданих, 21 сценарій, E1/E2/E3

## Для чого

Це нова, самостійна версія програми для вашого JupyterLab. Її ключова відмінність від v1/v2 — **мітки та групи походження отримуються з AUA-метаданих**, а не з назви папки. Експертне оцінювання **навмисно НЕ включено**.

Вхідні дані зберігаються на сервері за шляхом `/home/jovyan/work/audio_real_fake/AUA`. Архів містить код, **не містить 12+ годин аудіо**, не змінює вихідні дані.

## Склад архіву

- `SAADE_AUA_v3_1_Full_Pipeline.ipynb` — основний повний сценарій дослідження (34 комірки).
- `SAADE_AUA_v3_1_Results_Viewer.ipynb` — окремий переглядач/прослуховування результатів без повторного навчання.
- `saade_v3_data.py` — зіставлення AUA-метаданих, визначення міток, груп, формування первинних split.
- `saade_v3_core.py` — 21 акустичний сценарій, нейромережі, навчання, EER/ROC-AUC/F1/ASRI, bootstrap, freeze та ZIP.
- `saade_v3_methods.py` — сегментація після групового split, 3–5 seeds, GEE та захищений hidden.
- `saade_v3_visuals.py` — 21 аудіоплеєр, осцилограми, спектрограми, heatmaps, криві, матриці помилок.
- `requirements_v3.txt`, `test_smoke.py` — пакети та мінімальний тест на штучній структурі.

**Потрібно розпакувати всі чотири модулі `.py` в ту саму папку, що й ноутбуки.** Старі модулі `saade_core.py` / `saade_visuals.py` v1/v2 НЕ використовуються.

## Перший запуск

1. Завантажте архів у JupyterLab. У терміналі Jupyter перейдіть у папку із ZIP та розпакуйте його (або розпакуйте на комп'ютері та завантажте файли окремо).
2. У разі відсутніх пакетів у новій комірці Jupyter виконайте:

   ```python
   %pip install -r requirements_v3.txt
   ```

   Якщо PyTorch і CUDA уже працюють, **не перевстановлюйте `torch`** без потреби. Перевірте окремо `ffmpeg -version`.
3. Відкрийте `SAADE_AUA_v3_1_Full_Pipeline.ipynb`.
4. У першій кодовій комірці залиште:

   ```python
   DATA_ROOT = Path('/home/jovyan/work/audio_real_fake/AUA')
   RESULTS_ROOT = Path('/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v3_1_metadata_fixed')
   EXPERIMENT_MODE = 'pilot'
   RUN_TRAINING = True
   RUN_REPEATS = False
   RUN_FINAL_HIDDEN_TEST = False
   ```
5. `Kernel → Restart Kernel and Run All Cells`.
6. Знайдіть результат: `/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v3_1_metadata_fixed/SAADE_RESULTS.zip`. Завантажте ZIP сюди для перевірки.

Відкривши Viewer, вкажіть той самий `RESULTS_ROOT`. Плеєри працюють у Jupyter з локальних аудіофайлів AUA; не вимагають завантаження звуку до ChatGPT.

## Що саме змінено у підключенні AUA

- `CommonVoice` (REAL) — первинний каталог AUA + `speaker_id_basis=common_voice_client_id`.
- `AUA_own` (REAL) — `01_audio/own_processed/metadata_audio.csv`, оригінальний запис, справжній `speaker_id`.
- `FLEURS` (REAL за походженням корпусу) — технічний `S1108` **не приймається за реального мовця**, якщо `source_group_unknown`; записи потрапляють лише до TRAIN до відновлення ідентичностей.
- `DRSSU` (FAKE) — `generated=True` з `00_sources_and_licenses/audio/drssu/metadata.csv`, зіставлене за `source_record_id=drssu:<selection_rank>`; технічний `S1109` **не приймається за ідентифікатор голосу**. Невідомі голоси потрапляють лише до TRAIN.
- `AUA_new_TTS` (FAKE) — лише записи, які зіставляються з `metadata_new-tts.csv` та мають `model_name` і `voice_id`. Групування за генератором+голосом, а не за filename.

**Статус `approval_status=confirmed` НЕ означає, що достовірність звуку перевірено експертами.** У цьому дослідженні мітки позначають походження записів.

## Методика

1. Зіставлення `working_path`, `file_name`, `source_record_id`, хешів та оригінальних таблиць.
2. Початкові `train/validation/hidden_test` — лише за підтвердженими speaker/voice-групами. Невідомі групи можуть бути **TRAIN ONLY**, не враховуються як незалежні test-групи.
3. Нарізання **неперекривних 4-секундних** фрагментів уже після поділу.
4. **21 сценарій:** 1 baseline + 5 рівнів кожної з чотирьох категорій:

   | Сценарій | Рівень 1 | 2 | 3 | 4 | 5 |
   | --- | --- | --- | --- | --- | --- |
   | Noise SNR, dB | 20 | 15 | 10 | 5 | 0 |
   | MP3, kbps | 64 | 48 | 32 | 24 | 16 |
   | Синтетична реверберація, номінальний RT60, с | 0.2 | 0.35 | 0.5 | 0.7 | 0.9 |
   | Low-pass, Hz | 7000 | 6000 | 5000 | 4200 | 3400 |

5. **E1 Static** без аугментації; **E2 Random** з випадковим вибором категорій; **E3 Adaptive** з вагами слабких категорій на validation.
6. Для кожного `модель × режим × сценарій` — BA, EER, ROC-AUC, F1, false alarm/miss, ASRI, зміна від baseline; 21 аудіоплеєр і порівняльні графіки.
7. Bootstrap за групами, GEE з взаємодіями, повтори за 3–5 seeds після активації `RUN_REPEATS=True`.
8. Hidden test **не запускається** за замовчуванням, доки не буде доказового source-matched корпусу, завершених повторів і зафіксованого протоколу.

## Які моделі

У першому запуску: `Spectrogram CNN`, `Raw Wave CNN`, `MFCC-SVM`. **Це базові моделі, не AASIST і не RawNet2**.

Для додаткових архітектур після підготовки репозитаріїв та залежностей змініть `MODELS` на `('spectrogram_cnn','raw_wave_cnn','mfcc_svm','aasist','wavlm')` або включіть `rawnet2`. Ці моделі підтримані адаптерами, але **не перевірялися у вашому середовищі**.

- AASIST: [офіційний код](https://github.com/clovaai/aasist), розташування `/home/jovyan/work/aasist`; `models/AASIST.py`, `config/AASIST.conf`.
- RawNet2: [ASVspoof5](https://github.com/asvspoof-challenge/asvspoof5), розташування `/home/jovyan/work/asvspoof5`; `Baseline-RawNet2/model.py`.
- WavLM: `transformers` + checkpoint `microsoft/wavlm-base-plus`; pretrained weights завантажуються через Hugging Face (потрібен доступ до Інтернету або локальний кеш).

Для порівняння режимів використовуйте однакові `MODELS`, `SEEDS`, кількість епох і аудіосегменти. При зміні параметрів — **нова папка `RESULTS_ROOT`**.

## Принципові обмеження, які не можна виправити програмно

У поточному AUA всі реальні записи походять із одних джерел, а всі синтетичні — з інших. **Джерело і клас повністю змішані.** Модель може навчитися розпізнавати студію, кодек або корпус, а не підробку. Жоден автоматичний «баланс за класами» цього не виправляє. У `EXPERIMENT_MODE='research'` код **зупиниться**, доки не з'явиться хоча б одне джерело з обома класами. Навіть після цього потрібен контроль зіставності записів/голосів/каналів.

Ні STFT, ні сценарні графіки, ні EER не є **експертною валідацією**. У v3 її не реалізовано, як ви і просили.

## Перевірка та статус

- Синтаксис `.py` і формат `.ipynb` перевірено.
- Повний ноутбук перевірено **від початку до кінця на невеликому тестовому AUA з одним MFCC-SVM**.
- Окремо перевірено три режими E1/E2/E3, повноту всіх 21 сценарію, групування, аудіоперетворення, експорт.
- **Повний запуск із реальним набором AUA, 3 CNN та GPU ще не проведений.** Не стверджуйте, що наукова точність або source matching підтверджені до запуску на сервері.

## Вміст архіву v3.1

- `SAADE_AUA_v3_1_Preflight_Data_Audit.ipynb` — аудит походження, зв’язків DRSSU, придатності до навчання без навчання моделей.
- `SAADE_AUA_v3_1_Full_Pipeline.ipynb` — повний дослідницький конвеєр, 21 сценарій, E1/E2/E3 і збереження метрик.
- `SAADE_AUA_v3_1_Results_Viewer.ipynb` — графіки, інтерактивні звукові приклади, всі сценарії та рішення моделей без перенавчання.
- `saade_v3_core.py`, `saade_v3_data.py`, `saade_v3_methods.py`, `saade_v3_visuals.py` — модулі, усі одночасно мають лежати поруч із ноутбуками.
- `requirements_v3.txt`, `PATHS_LOCK.json`, `test_drssu_adapter.py`, `test_smoke.py`, `MANIFEST_SHA256.json`.

Порядок: спочатку preflight → перевірка прив’язки DRSSU → повний pipeline → viewer. У JupyterLab зробіть **Kernel → Restart Kernel** після заміни будь-якого модуля. Не змішуйте зі старим v3.0.
