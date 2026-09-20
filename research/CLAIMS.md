# دفتر شواهد برای مقاله

این دفتر مشخص می‌کند چه چیزی با کدام شاهد قابل گفتن است. «مشاهدهٔ اکتشافی»، «پذیرش مهندسی»، «کالیبراسیون کمکی» و «تأیید مستقل پاسخ نهایی» سطوح متفاوت‌اند. CL-001 تا CL-007 شواهد تاریخی‌اند؛ CL-008 اعتبارسنجی زیرساخت است. آزمایش علمی جدیدی اجرا نشده است. مرور محدودیت‌ها در [roadmap](../docs/OPERATIONAL_ROADMAP_FA.md) ثبت شده است.

فهرست تولیدشدهٔ runها در [paper/RUN_CATALOG.md](paper/RUN_CATALOG.md) و جدول ماشینی در [paper/evidence.csv](paper/evidence.csv) است. [export_research_catalog.py](../scripts/export_research_catalog.py) آن‌ها را از رکوردهای پژوهش می‌سازد؛ فهرست خودکار جای تفسیر علمی و بررسی استقلال شاهد را نمی‌گیرد.

| شناسهٔ ادعا | جملهٔ مجاز و دامنه | شاهد مستقیم | محدودیت/شاهد بعدی |
|---|---|---|---|
| `CL-001` | مرجع تک‌بذر برای توسعه تثبیت و ممیزی شده است | [trained_m0_audit](../research/legacy/LEGACY-M0/manifests/trained_m0_audit.json)، [گزارش](../research/legacy/LEGACY-M0/reports/M0_TRAINED_REPORT_FA.md) | بازتولید کامل چهاربذر مقاله و ادعای برتری نهایی تأیید نشده‌اند |
| `CL-002` | هستهٔ sparse phase آزمون‌های ثبت‌شدهٔ M1 را پاس کرده است | [m1_status](../research/legacy/LEGACY-M1/manifests/m1_status.json)، [گزارش](../research/legacy/LEGACY-M1/reports/M1_REPORT_FA.md) | دامنهٔ ثابتِ مسئلهٔ داخلی؛ پایداری حلقهٔ coupled نتیجه نمی‌شود |
| `CL-003` | تلاش‌های اولیهٔ anchor معیار یادگیری M2 را پاس نکردند | [m2_status](../research/legacy/LEGACY-M2-CONTEXT/manifests/m2_status.json)، [m2_diagnostic](../research/legacy/LEGACY-M2-DIAGNOSTIC/manifests/m2_diagnostic.json) | شاهد منفی باید همراه شاخهٔ موفق‌تر باقی بماند؛ همهٔ معماری‌های ممکن رد نشده‌اند |
| `CL-004` | probe روی پیشنهادهای native در monitor ثبت‌شده دقت متوازن ۱۰۰٪ داشت؛ کنترل تصادفی نزدیک شانس بود | [m2_proposal](../research/legacy/LEGACY-M2-PROPOSAL/manifests/m2_proposal.json) | این دقت پاسخ نهایی نیست؛ مدل native cohort را دیده و monitor در توسعه استفاده شده است |
| `CL-005` | M2-Q روی monitor توسعه دقت متوازن ۸۱٫۲۱۶٪ با کنترل ۳۲٫۱۸۷٪ ثبت کرد | [m2_q_calibration](../research/legacy/LEGACY-M2-Q-CALIBRATION/manifests/m2_q_calibration.json) | نتیجهٔ کمکی و اکتشافی؛ checkpoint منتخب ذخیره نشده؛ نسخهٔ اصلاح‌شدهٔ معماری است؛ تأیید تازه لازم است |
| `CL-006` | Q-shadow محاسبات detached و نرمال‌سازی mixture را روی نمونه‌های ثبت‌شده بررسی کرد | [m2_q_shadow](../research/legacy/LEGACY-M2-Q-SHADOW/manifests/m2_q_shadow.json) | مقایسهٔ قبل/بعد native روی مثال محدود و خارج از حلقهٔ coupled است؛ zero-effect و علیت اتصال واقعی تأیید نشده‌اند |
| `CL-007` | اثر ERGT-Phi بر پاسخ نهایی، تعمیم مستقل و هزینهٔ کامل هنوز نتیجه‌گیری نشده است | [roadmap](../docs/OPERATIONAL_ROADMAP_FA.md)، [STATE](STATE.md) | نیازمند M3/M4 و ارزیابی قفل‌شدهٔ مستقل M8 است |
| `CL-008` | چرخهٔ ساخت بسته، اجرای محلی، ورود نتیجه و بازبینی نسخه‌دار روی نمونهٔ WF-E001 کار کرده است | [تفسیر run](experiments/WF-E001/v001/runs/local-20260920T111545Z-22c8c9/interpretation.md)، [اعتبارسنجی ۳۶ تست](validation/WORKFLOW_V1.md)، [ADR-0002](decisions/ADR-0002.md) | شاهد مدیریت آزمایش؛ صحت اجرای زندهٔ Colab و هر بهبود علمی مدل هنوز از این نمونه نتیجه نمی‌شود |

برای رکوردهای legacy، hashهای موجود در manifest اصلی و فهرست [legacy/index.json](legacy/index.json) منبع‌اند. ادعای وجود رکورد بازبینی جدید، hash بازبینی یا provenance جدید برای این اجراها ساخته نشود.

## قالب ادعای جدید

هر ادعای تازه باید شناسه، جملهٔ دقیق، سطح شاهد، نسبت با سند/فرضیه، آزمایش و نسخه و run، مسیر `summary.md`، `interpretation.md` و رکورد دقیق `reviews/NNN.json`، hash مدارک اصلی، پروتکل و snapshot را داشته باشد. همچنین این موارد لازم‌اند:

- seedها و تعداد اجرا/نمونه، روش محاسبه و پراکندگی یا عدم قطعیت؛
- منشأ split و تاریخچهٔ استفاده از آن؛ استقلال واقعی از توسعه؛
- baseline و کنترل، بودجهٔ مقایسه و هزینهٔ کامل شامل گذر مقدماتی؛
- runهای منفی/متعارض، موارد حذف و دلیل ازپیش‌ثبت‌شدهٔ حذف؛
- محدودیت، دامنهٔ تعمیم، نمودار/جدول قابل بازتولید و فرمان تولید آن؛
- داوری مرتبط و ADR؛ عبارت مجاز در مقاله و ادعایی که این شاهد پشتیبانی نمی‌کند.

یک run می‌تواند معیار فنی را پاس کند و همچنان برای ادعای علمی کافی نباشد. تبدیل ادعای اکتشافی به تأییدی باید با دادهٔ بازنشده، معیار قفل‌شده و اجرای تازه مستند شود. با اصلاح استراتژی، ادعای قدیمی حذف نشود؛ وضعیت آن و پیوند شاهد جدید افزوده شود.
