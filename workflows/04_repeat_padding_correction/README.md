# SAADE–AUA v9.1 — повна корекція padding

## Для чого
Оновлений окремий Jupyter Notebook проводить **повний повторний інференс AASIST та AASIST-L** на всіх відкритих сегментах v9 з повторенням активного аудіосигналу до 64 600 семплів замість додавання 600 нулів (або до більшої активної частини у разі коротких сегментів). Результати Wav2Vec2 із v9 переносить без нового інференсу.

## Передумови
- Завершений v9 за `/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v9_COMPLETE_EVALUATION/FULL_EXPERIMENT/`, із SQLite або CSV прогнозів, метриками й протоколом.
- Присутні ONNX-файли v9 з незмінними SHA-256 у `SAADE_AUA_2026_v9_COMPLETE_EVALUATION/pretrained_model_cache/`.
- AUA `/home/jovyan/work/audio_real_fake/AUA` і каталог v6.2 із `V6_LOCKED_MATCHED_SPLITS.csv`.

## Запуск
1. Завантажте один `.ipynb` у JupyterLab.
2. **Kernel → Restart Kernel and Run All Cells.** У конфігурації вже `RUN_PHASE='FULL'`, без лімітів.
3. Результати з'являться в новій папці `results/SAADE_AUA_2026_v9_1_AASIST_REPEAT_PADDING/FULL_EXPERIMENT/`.
4. Можна перезапустити після збою: SQLite зберігає всі завершені сегменти.
5. Надішліть `results/SAADE_AUA_v9_1_REPEAT_PADDING_FULL_FOR_REVIEW.zip`.

## Що не зміниться
Оригінальний AUA, вміст старого v9 і Melnyk_dus. Hidden не відкривається.

## Важливі наукові уточнення
- Виправлення внесено після перегляду v9 validation: це **post-hoc** корекція, її слід відобразити в методології.
- 21 сценарій та шумові seed залишаються незмінними. Результати Wav2Vec2 перевіряються на 0 розбіжностей.
- У validation залишається лише один TTS-голос; тест не доводить узагальнення на невідомі генератори.
- Перевірено синтаксис/структуру коду; обчислювально важкий запуск реальних ONNX на віддаленому сервері не виконувався.
