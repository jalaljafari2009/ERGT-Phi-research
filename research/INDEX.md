# نقشهٔ پژوهش و مسیر ادامه

نقطهٔ شروع هر جلسه [STATE](STATE.md) است. سپس [registry.json](registry.json) و
`python -B scripts/research.py status` بررسی شوند. این نقشه هم شواهد قدیمی و هم
آزمایش‌های جدید را پوشش می‌دهد؛ برای پیدا کردن اطلاعات از روی حافظهٔ گفتگو حدس نزنید.

## هدف، برنامه و تصمیم

| اطلاعات | مسیر ثابت |
|---|---|
| هدف ریاضی | [docs/MATHEMATICAL_SPEC.md](../docs/MATHEMATICAL_SPEC.md) |
| قفل مبنای ریاضی | [specification.lock.json](specification.lock.json)، [بررسی](../scripts/check_spec_lock.py)، [حفاظت ویندوز](../scripts/protect_spec.ps1) |
| تصمیم برنامهٔ دو‌مسیره | [ADR-0004](decisions/ADR-0004.md)، [منشأ گفتگو](sources/chat/20260920_DUAL_TRACK_DECISION_FA.md) |
| برنامهٔ فعال | [docs/OPERATIONAL_ROADMAP_FA.md](../docs/OPERATIONAL_ROADMAP_FA.md) |
| قرارداد فعال M2-Q v2 | [افزونهٔ Q-anchor](../docs/SPEC_Q_ANCHOR_ADDENDUM.md)، [config](../configs/m2_q_v2.json)، [ADR-0006](decisions/ADR-0006.md)، [اعتبارسنجی](validation/M2_Q_V2_CONTRACT_20260921.md)؛ طراحی تثبیت‌شده، بدون run تازه |
| آزمایش تحویل checkpoint M2-Q | [M2-E002/v001](experiments/M2-E002/v001/README.md)، [اعتبارسنجی runner](validation/M2_E002_RUNNER_20260921.md)؛ کد و نوت‌بوک آماده، اجرای واردشده هنوز موجود نیست |
| فرضیهٔ دوطرفه و آزمایش‌های چهار بازو | [طرح B0/B1 نسخهٔ ۰۰۱](plans/BIDIRECTIONAL_BOUNDARY_V001_FA.md)، [ADR-0005](decisions/ADR-0005.md)، [منشأ گفتگو](sources/chat/20260920_BIDIRECTIONAL_HYPOTHESIS_FA.md)؛ طراحی، بدون اجرای علمی |
| هدف نسخه‌دار و بازبینی فاز | [goals.json](goals.json)، [M2_CURRENT](phase_reviews/M2_CURRENT.md) |
| بازبینی مسیر با اولویت هدف علمی | [MISSION_REVIEW_20260920](phase_reviews/MISSION_REVIEW_20260920.md)؛ یافته‌های کد و پیشنهاد آزمایش اثرگذاری بر پاسخ، هنوز تصمیم تازهٔ اجرا نیست |
| وظایف ایجنت و چرخهٔ پژوهش | [AGENTS](../AGENTS.md)، [WORKFLOW](WORKFLOW.md) |
| مسیرها و تصمیم‌های آزموده‌شده | [ARCHITECTURE](ARCHITECTURE.md)، [DECISIONS](DECISIONS.md) |
| ترتیب پیشرفت‌ها | [progress.jsonl](progress.jsonl) |

## منابع و اطلاعات گفتگو

[sources/README.md](sources/README.md) ورودی همهٔ منابع است:
[فهرست با هش](sources/register.json)، [تاریخچهٔ گفتگو](sources/CONVERSATION_HISTORY_FA.md)،
[نیازمندی‌های پژوهشگر](sources/RESEARCHER_REQUIREMENTS_FA.md)،
[خروجی‌های نقل‌شدهٔ ترمینال](sources/chat/terminal_excerpts.md)، مقاله، spec اولیه و متن‌های پیوست.
نسخهٔ مقالهٔ ارسالی با PDF داخل مرجع متفاوت است؛ تفاوت ثبت شده و هنوز تطبیق علمی نشده است.

## آزمایش‌های گذشته، همراه نتیجه و مسیر بعد

| پرونده | نقش در مسیر پژوهش |
|---|---|
| [LEGACY-M0](legacy/LEGACY-M0/README.md) | مهندسی اولیه، Colab تک‌بذر، checkpoint و ممیزی توسعه |
| [LEGACY-M1](legacy/LEGACY-M1/README.md) | هستهٔ مستقل؛ هر دو اجرای ثبت‌شدهٔ آزمون‌ها |
| [LEGACY-M2-INITIAL](legacy/LEGACY-M2-INITIAL/README.md) | شکست anchor اولیه؛ manifest/protocol بازیابی‌شده از Git |
| [LEGACY-M2-CONTEXT](legacy/LEGACY-M2-CONTEXT/README.md) | اصلاح زمینه‌دار و ممیزی حساسیت؛ کالیبراسیون ناکافی |
| [LEGACY-M2-DIAGNOSTIC](legacy/LEGACY-M2-DIAGNOSTIC/README.md) | مقایسهٔ معماری‌های token/context/pair و کنترل تصادفی |
| [LEGACY-M2-INFORMATION](legacy/LEGACY-M2-INFORMATION/README.md) | سنجش اطلاعات میدان اولیه، نهایی و تاریخچه |
| [LEGACY-M2-PROPOSAL](legacy/LEGACY-M2-PROPOSAL/README.md) | probe پیشنهاد بومی و کنترل برچسب |
| [LEGACY-M2-Q-TIMING](legacy/LEGACY-M2-Q-TIMING/README.md) | قرارداد اولیهٔ Q تأخیری و حدود پیاده‌سازی |
| [LEGACY-M2-Q-SHADOW](legacy/LEGACY-M2-Q-SHADOW/README.md) | محاسبهٔ shadow جدا از حلقهٔ واقعی |
| [LEGACY-M2-Q-CALIBRATION](legacy/LEGACY-M2-Q-CALIBRATION/README.md) | معیار کمکی موفق، تحویل checkpoint ناقص |

فهرست ماشینی [legacy/index.json](legacy/index.json) و نقشهٔ نام قدیم به جدید
[legacy/path_map.json](legacy/path_map.json) هستند. هر پرونده، گزارش‌ها، manifestها،
شواهد کوچک، کد دارای منشأ Git و نوت‌بوک موجود را کنار هم نگه می‌دارد. گذشته به run
ازپیش‌ثبت‌شدهٔ جدید تبدیل نشده است. اطلاعات مفقود و اختلاف گزارش با خروجی در README
همان پرونده توضیح داده شده‌اند.

## آزمایش‌های جدید

[experiments/README.md](experiments/README.md) و [registry.json](registry.json) مسیر را تعیین می‌کنند:

```text
experiments/<id>/v001/
  protocol.json           سؤال، روش، ورودی‌ها، معیارها و parent
  experiment.ipynb        نوت‌بوک منبع همان نسخه
  notebook.sha256         hash نوت‌بوک
  package.json            قفل بستهٔ سورس و commit
  source_manifest.json    inventory دقیق منبع
  README.md               راهنما و پیوند همهٔ runهای نسخه
  runs/<run_id>/
    run.json / artifacts.json / evaluation.json / summary.md
    interpretation.md / reviews/001.json / files/
```

نمونهٔ پذیرفته‌شدهٔ زیرساخت: [WF-E001/v001](experiments/WF-E001/v001/README.md).

## مقاله، آرشیو و بازیابی

- [paper/README.md](paper/README.md): جدول آزمایش‌های تاریخی و جدید، CSV معیارها و نحوهٔ استفاده.
- [CLAIMS](CLAIMS.md): عبارت مجاز علمی و محدودیت هر شاهد.
- `artifacts/legacy/<id>/`: نسخهٔ کامل محلی اجرای تاریخی، از جمله وزن‌ها؛ خارج از Git.
- `artifacts/source_archives/`: ZIPهای اصلی معرفی‌شده در گفتگو؛ locator و هش در منابع.
- `inbox/` و `packages/`: ورودی نتایج و خروجی بسته‌بندی؛ خارج از Git.
- [migrations](migrations/README.md): مسیر قبلی/جدید، هش‌ها، پاک‌سازی پوشه‌های خالی و حدود بازیابی.
- [گزارش سامان‌دهی](validation/REORGANIZATION_20260920.md): شمار مدارک و کنترل صحت مهاجرت.

فایل‌های بزرگ در clone تازه خودکار حاضر نیستند؛ موجودی هر پرونده و فهرست منابع
محل بازیابی را مشخص می‌کنند. وجود locator محلی، وجود نسخهٔ پشتیبان روی Drive را اثبات نمی‌کند.
