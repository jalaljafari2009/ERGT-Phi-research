# ERGT-Phi — فضای پژوهش

هدف پروژه، پیاده‌سازی و ارزیابی شاخهٔ فاز طبق [سند ریاضی](docs/MATHEMATICAL_SPEC.md)
است. مسیر اجرا در [برنامهٔ عملیاتی](docs/OPERATIONAL_ROADMAP_FA.md) و نقطهٔ ادامه در
[research/STATE.md](research/STATE.md) ثبت می‌شود.

**وضعیت علمی:** M0 مرجع توسعهٔ تک‌بذر دارد؛ M1 هستهٔ مستقل را گذرانده است.
M2-Q معیار کالیبراسیون کمکی را پاس کرده، اما checkpoint منتخب و تحویل اجرایی کامل
ندارد. اقدام بعدی تکمیل M2 است؛ M3 هنوز باز نشده است.

## مسیرهای اصلی

| نیاز | از اینجا شروع کنید |
|---|---|
| راهنمای ایجنت و پیگیری کارها | [AGENTS.md](AGENTS.md) |
| نقشهٔ کامل پروژه | [research/INDEX.md](research/INDEX.md) |
| اطلاعات گفتگو، پیوست‌ها و نیازمندی‌های پژوهشگر | [research/sources/README.md](research/sources/README.md) |
| تمام آزمایش‌های پیشین، شکست‌ها و اصلاحات | [research/legacy/README.md](research/legacy/README.md) |
| آزمایش‌های دارای پروتکل و نوت‌بوک نسخه‌دار | [research/experiments/README.md](research/experiments/README.md) |
| معماری، راه‌های جایگزین و تصمیم‌ها | [ARCHITECTURE](research/ARCHITECTURE.md)، [DECISIONS](research/DECISIONS.md) |
| شواهد قابل استفاده در مقاله | [دفتر ادعاها](research/CLAIMS.md)، [راهنمای مقاله](research/paper/README.md) |

## ساختار

```text
docs/                  سند ریاضی و برنامهٔ فعال
reference/             مرجع ثابت مقاله و داده‌های ثبت‌شدهٔ آن
ergt_phi/              پیاده‌سازی پژوهشی
scripts/               ابزارهای اجرا، ممیزی و مدیریت پژوهش
tests/                 آزمون‌ها
configs/               تنظیمات
research/
  sources/             مقاله، spec اولیه، متن‌های پیوست و اطلاعات گفتگو
  legacy/              ده پروندهٔ تاریخی، هرکدام با گزارش و شواهد خودش
  experiments/         آزمایش‌های جدید با نسخهٔ نوت‌بوک و runهای متصل
  decisions/           دلیل انتخاب، اصلاح یا کنارگذاشتن مسیرها
  phase_reviews/       مرور هدف و مدارک در پایان فاز
  paper/               جدول شواهد و ادعاهای قابل گزارش
  migrations/          نقشه و هش جابه‌جایی‌ها
  validation/          گزارش بررسی زیرساخت و سامان‌دهی
  artifacts/           وزن‌ها و آرشیوهای محلی؛ خارج از Git
  packages/            ZIPهای ورودی Colab؛ خارج از Git
  inbox/               خروجی‌های دریافتی برای بررسی؛ خارج از Git
  workspace/           فایل‌های موقت ابزارهای قدیمی؛ خارج از Git
```

`runs/` محل scratch اجرای ابزارهای قدیمی و `.cache/` محل تست‌های موقت است؛
شاهد پذیرفته‌شده باید وارد پروندهٔ آزمایش شود. نتایج تاریخی از این پوشه‌های
پراکنده به `research/legacy/` و `research/artifacts/legacy/` منتقل شده‌اند.

## ادامهٔ کار

ایجنت ابتدا STATE و وضعیت ابزار را می‌خواند، نتیجهٔ رسیده را بررسی می‌کند و
با توجه به برنامه و آخرین تصمیم ادامه می‌دهد:

```powershell
python -B scripts/research.py status
```

دستورهای ساخت نسخه/نوت‌بوک، ورود ZIP و ثبت داوری در [WORKFLOW](research/WORKFLOW.md)
آمده‌اند. یک نمونهٔ کاملِ محلی در [WF-E001/v001](research/experiments/WF-E001/v001/README.md)
ثبت است. ساخت نوت‌بوک به معنی شروع خودکار Colab نیست.

## اجرای ابزارها و نصب محیط

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-m0.lock.txt
.venv/Scripts/python.exe -B scripts/research.py --help
```

دستور را در ترمینال PowerShell اجرا کنید؛ علامت `>>>` یعنی داخل مفسر Python هستید.
فایل lock، محیط محلی ثبت‌شده را توصیف می‌کند؛ نوت‌بوک علمی باید محیط و نیاز GPU خودش
را مشخص کند. راهنمای ابزارهای تاریخی در [scripts/README.md](scripts/README.md) است.

کد `reference/` و مجوز آن ثابت مانده‌اند. شرح پراکندهٔ قبلی README با بایت‌های
اصلی در [snapshot پیش از سامان‌دهی](research/legacy/workspace_snapshots/10f1bd4/README.original.md.txt)
محفوظ است. [گزارش سامان‌دهی](research/validation/REORGANIZATION_20260920.md) جزئیات انتقال را دارد.
