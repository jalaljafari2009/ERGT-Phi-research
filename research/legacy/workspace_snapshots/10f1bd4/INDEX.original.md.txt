# نقشهٔ ثابت پژوهش

برای ادامهٔ کار از [STATE](STATE.md) و [registry.json](registry.json) شروع کنید. registry فهرست ماشینی آزمایش‌ها و مسیر نسخه‌هاست؛ وضعیت runها از مدارک داخل هر نسخه خوانده می‌شود. پروتکل قفل‌شده، bundle و گزارش بازبینی شاهد نتیجه‌اند. STATE خلاصهٔ انسانی را نگه می‌دارد. در صورت اختلاف، شاهد را بررسی و یک رویداد اصلاحی ثبت کنید؛ سکوت یا بازنویسی تاریخچه راه حل نیست.

| پرسش | فایل |
|---|---|
| ایجنت چه کارهایی را خودکار پیگیری می‌کند؟ | [AGENTS.md](../AGENTS.md) |
| هدف و قرارداد ریاضی چیست؟ | [MATHEMATICAL_SPEC.md](../docs/MATHEMATICAL_SPEC.md) |
| ترتیب فازها و معیارهای خروج چیست؟ | [OPERATIONAL_ROADMAP_FA.md](../docs/OPERATIONAL_ROADMAP_FA.md) |
| چرخهٔ نوت‌بوک تا نتیجه چگونه است؟ | [WORKFLOW.md](WORKFLOW.md) |
| اکنون چه چیزی باز است؟ | [STATE.md](STATE.md) |
| هدف‌های نسخه‌دار و دامنهٔ فعال چیست؟ | [goals.json](goals.json) |
| بازبینی جاری M2 کجاست؟ | [phase_reviews/M2_CURRENT.md](phase_reviews/M2_CURRENT.md) |
| معماری جاری و مسیرهای جایگزین چیست؟ | [ARCHITECTURE.md](ARCHITECTURE.md) |
| چرا تصمیم‌ها گرفته شدند؟ | [DECISIONS.md](DECISIONS.md) |
| چه ادعایی برای مقاله مجاز است؟ | [CLAIMS.md](CLAIMS.md) |
| جدول همهٔ اجراها برای مقاله کجاست؟ | [paper/RUN_CATALOG.md](paper/RUN_CATALOG.md)، [paper/evidence.csv](paper/evidence.csv)؛ تولید با [export_research_catalog.py](../scripts/export_research_catalog.py) |
| ترتیب رویدادهای ثبت‌شده چیست؟ | [progress.jsonl](progress.jsonl) |
| آزمایش/نسخه/run کجاست؟ | [registry.json](registry.json) و [experiments/](experiments/) |
| متن مستقل تصمیم کجاست؟ | `research/decisions/ADR-####.md` از [DECISIONS](DECISIONS.md) |
| فهرست ماشینی مدارک قدیمی کجاست؟ | [legacy/index.json](legacy/index.json) |
| ابزار چرخه کجاست؟ | [scripts/research.py](../scripts/research.py) |
| قالب‌های مدارک کجاست؟ | [پروتکل](templates/protocol.json)، [ADR](templates/decision.md)، [بازبینی فاز](templates/phase_review.md)، [تفسیر علمی](templates/interpretation.md) |

ساختار جدید هر نوت‌بوک:

```text
research/experiments/<experiment_id>/
  v001/
    protocol.json          # سؤال، روش، معیار، parent و فرمان
    experiment.ipynb       # نوت‌بوک منبع برای Colab
    notebook.sha256        # هویت نوت‌بوک منبع
    package.json           # snapshot و قفل منبع بسته‌شده
    source_manifest.json   # فهرست hash تمام فایل‌های داخل بسته
    README.md              # توضیح و پیوند به تمام اجراهای این نسخه
    runs/<run_id>/
      run.json             # هویت و محیط اجرای واردشده
      artifacts.json       # hash، اندازه و مکان خروجی‌ها
      summary.md           # خلاصهٔ ماشینیِ محفوظِ معیارهای run
      evaluation.json      # نتیجهٔ ماشینی معیارهای قفل‌شده
      interpretation.md    # تحلیل علمی، محدودیت و پیشنهاد ایجنت
      reviews/001.json     # داوری و پیوند ADR؛ اصلاح در رکورد بعدی
      files/               # فایل‌های سبک شاهد
  v002/                    # اصلاح پروتکل/کد علمی؛ نسخهٔ قبل محفوظ
```

`research/inbox/` محل ZIP بازگشتی، `research/packages/` محل بستهٔ آپلود و `research/artifacts/` محل محلی فایل‌های حجیم هستند. این سه مسیر در Git ثبت نمی‌شوند؛ hash، اندازه و مکان بازیابی artifact باید در مدارک tracked موجود باشد. برای نگهداری بلندمدت وزن‌ها یک نسخهٔ ماندگار مانند Drive همراه با شناسه و hash لازم است؛ وجود یک مسیر محلی به‌تنهایی نسخهٔ پشتیبان نیست.

`source_manifest.json` و `package.json` همراه نسخه در Git می‌مانند. در صورت حذف ZIP سورس، فرمان `package` همان ZIP را فقط وقتی بازسازی می‌کند که فایل‌های ورودی دقیقاً با inventory مطابق باشند. `source_commit` در بستهٔ ساخته‌شده از working tree غیرتمیز فقط HEAD پایه است و کافی‌بودن آن برای بازیابی نباید فرض شود. برای بستهٔ علمی، ابتدا source/protocol/notebook را commit کنید؛ بایگانی مستقل ZIP با hash نیز راه بازیابی مستقیم است.

## شواهد پیش از استقرار این ساختار

این شواهد جابه‌جا یا به اجرای جدید تبدیل نشده‌اند. هنگام مقایسه، آن‌ها را `legacy` بخوانید و به مسیر اصلی ارجاع دهید. فهرست ماشینی کامل‌تر در [legacy/index.json](legacy/index.json) است؛ جدول زیر راهنمای سریع است.

| مرحله/مسیر | گزارش انسانی | مدرک ماشینی یا نوت‌بوک |
|---|---|---|
| M0 و مرجع توسعه | [M0_REPORT_FA](../docs/M0_REPORT_FA.md)، [M0_TRAINED_REPORT_FA](../docs/M0_TRAINED_REPORT_FA.md) | [m0_status](../manifests/m0_status.json)، [trained_m0_audit](../manifests/trained_m0_audit.json)، [نوت‌بوک قدیمی M0](../notebook/M0_Reference_Training.ipynb) |
| M1 | [M1_REPORT_FA](../docs/M1_REPORT_FA.md) | [m1_status](../manifests/m1_status.json) |
| M2 anchor اصلی | [M2_REPORT_FA](../docs/M2_REPORT_FA.md) | [m2_status](../manifests/m2_status.json) |
| M2 anchor زمینه‌دار | [M2_CONTEXTUAL_REVISION_FA](../docs/M2_CONTEXTUAL_REVISION_FA.md) | [anchor_context_audit](../manifests/anchor_context_audit.json) |
| تشخیص معماری جفتی | [M2_DIAGNOSTIC_FA](../docs/M2_DIAGNOSTIC_FA.md) | [m2_diagnostic](../manifests/m2_diagnostic.json) |
| probe میدان و تاریخچه | [M2_INFORMATION_PROBE_FA](../docs/M2_INFORMATION_PROBE_FA.md) | [m2_information](../manifests/m2_information.json) |
| probe پیشنهاد بومی | [M2_PROPOSAL_PROBE_FA](../docs/M2_PROPOSAL_PROBE_FA.md) | [m2_proposal](../manifests/m2_proposal.json) |
| Q تأخیری و shadow جدا | [M2_LAGGED_Q_FA](../docs/M2_LAGGED_Q_FA.md)، [M2_Q_SHADOW_FA](../docs/M2_Q_SHADOW_FA.md) | [m2_q_shadow](../manifests/m2_q_shadow.json) |
| کالیبراسیون Q | [M2_Q_CALIBRATION_FA](../docs/M2_Q_CALIBRATION_FA.md) | [m2_q_calibration](../manifests/m2_q_calibration.json)، `runs/m2_q_calibration/` |

نتیجهٔ قدیمی Q دقت پایش کمکی را تأیید می‌کند؛ محدودیت checkpoint و اتصال واقعی در [STATE](STATE.md) و [CLAIMS](CLAIMS.md) صریح است. گزارش تاریخی که عبارت کلی‌تر دارد باید با این حدود تفسیر شود.
