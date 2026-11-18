# Як завантажити репозиторій на GitHub

1. Розпакуйте ZIP. Відкрийте папку `SAADE_AUA_GITHUB_COMPLETE_SOURCE`.
2. Створіть порожній репозиторій на GitHub (рекомендується спочатку **Private**).
3. У терміналі перейдіть до розпакованої папки та виконайте:

```bash
git init
git add .
git commit -m "Add SAADE-AUA research source code and version history"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Замініть `USERNAME/REPOSITORY`. Якщо репозиторій створений із README, спочатку синхронізуйте історію або створіть порожній репозиторій без README.

4. Перед відкриттям доступу додайте відповідну ліцензію та перевірте права на сторонні бібліотеки, код і моделі.

**Не завантажуйте ZIP як єдиний файл до кореня репозиторію**, якщо хочете, щоб код можна було переглядати в GitHub: розпакуйте його та завантажте вміст із збереженням папок.

## Що вилучено
Повні аудіокорпуси та TTS-пари, pretrained weights, результати інференсу, результати hidden test, особисті credentials, великі архіви звітів. Публічне посилання на набір: https://huggingface.co/datasets/adam-i/ATV-UA

## Захист hidden test
У `workflows/05_freeze_hidden_once/` початкові параметри безпечні (`FREEZE` / `NO`). Історичний ноутбук `historical/v9_2_ready_safe_template/` **навмисно повернуто до безпечних значень**; оригінальна версія була налаштована на одноразовий прихований запуск у конкретному серверному середовищі. Не використовуйте реальні hidden labels для подальшого підбору моделей.
