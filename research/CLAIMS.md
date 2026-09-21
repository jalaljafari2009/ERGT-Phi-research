# دفتر شواهد برای مقاله

این دفتر مشخص می‌کند چه چیزی با کدام شاهد قابل گفتن است. «مشاهدهٔ اکتشافی»، «پذیرش مهندسی»، «کالیبراسیون کمکی» و «تأیید مستقل پاسخ نهایی» سطوح متفاوت‌اند. CL-001 تا CL-007 شواهد تاریخی‌اند؛ CL-008 اعتبارسنجی زیرساخت، CL-009 تحویل مهندسی M2 و CL-010 تأیید توسعه‌ایِ همراه با نتیجهٔ منفی کنترل‌هاست. تأیید علمی تازهٔ پاسخ نهایی هنوز وجود ندارد. مرور محدودیت‌ها در [roadmap](../docs/OPERATIONAL_ROADMAP_FA.md) ثبت شده است.

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
| `CL-009` | release قفل‌شدهٔ M2-E002 در اجرای محلی package-bound چرخهٔ انتخاب نخستین checkpoint، freeze، reload، resume و حفظ مرجع را در هر ۱۹ gate پاس کرد | [خلاصه](experiments/M2-E002/v001/runs/20260921T210953Z-48abf488-local/summary.md)، [تفسیر](experiments/M2-E002/v001/runs/20260921T210953Z-48abf488-local/interpretation.md)، [بازبینی](experiments/M2-E002/v001/runs/20260921T210953Z-48abf488-local/reviews/001.json)، [ADR-0007](decisions/ADR-0007.md) | اجرای Windows/CPU و خارج از Colab؛ monitor دیده‌شده و کالیبراسیون کمکی است؛ توسعهٔ تازه، اثر بر پاسخ و مجوز M3 نتیجه نمی‌شوند |
| `CL-010` | checkpoint منجمد phase روی دو پنجرهٔ تازه دقت متوازن ۰٫۷۸۰ و ۰٫۷۷۱ داشت و هر ۲۲ gate را پاس کرد؛ Q-only و no-phase هم‌ظرفیت در هر دو پنجره ۱٫۰ بودند، پس سود اختصاصی phase روی target relation تأیید نشد | [خلاصه](experiments/M2-E003/v004/runs/20260921T224500Z-m2e003-v004-local/summary.md)، [تفسیر](experiments/M2-E003/v004/runs/20260921T224500Z-m2e003-v004-local/interpretation.md)، [بازبینی](experiments/M2-E003/v004/runs/20260921T224500Z-m2e003-v004-local/reviews/001.json)، [ADR-0011](decisions/ADR-0011.md) | توسعهٔ تازه و pair/text-disjoint است، اما target کمکی است؛ پاسخ native، علیت فاز، افق نهایی و مجوز M3 نتیجه نمی‌شوند |

برای رکوردهای legacy، hashهای موجود در manifest اصلی و فهرست [legacy/index.json](legacy/index.json) منبع‌اند. ادعای وجود رکورد بازبینی جدید، hash بازبینی یا provenance جدید برای این اجراها ساخته نشود.

## فرضیه‌های برنامه‌ریزی‌شده؛ فاقد شاهد اجرا

[طرح دوطرفهٔ B](plans/BIDIRECTIONAL_BOUNDARY_V001_FA.md) طبق
[ADR-0005](decisions/ADR-0005.md) سه پرسش برای آزمایش ثبت کرده است:

| شناسه | گزارهٔ قابل آزمون | وضعیت/شاهد لازم |
|---|---|---|
| `H-BD-D` | انتشار دوطرفه در بودجهٔ محدود سودی نسبت به انتشار یک‌طرفه دارد | planned/not_executed؛ مقایسهٔ هم‌اطلاع و کیفیت–هزینه با شاهد معتبر |
| `H-BD-P` | فاز فراتر از نگهداری چند فرضیهٔ مثبت سود مستقل دارد | planned/not_executed؛ کنترل مثبت و dephasing، ظرفیت و هزینهٔ متناظر |
| `H-BD-I` | جهت و فاز تعامل مفیدی دارند | planned/not_executed؛ اختلافِ اختلاف‌ها و عدم‌قطعیت، سپس تأیید مستقل |

این‌ها CL تأییدشده نیستند و به جدول نتایج مقاله عددی اضافه نمی‌کنند. همانی خطی دقیق
آزمون صحت است؛ برابری آن شاهد بهبود کیفیت نیست. پس از اجرای واقعی، هر فرضیه به
run/review و تفسیر مستقل وصل شود؛ تأیید یا رد یکی به دیگری تعمیم داده نشود.

## قالب ادعای جدید

هر ادعای تازه باید شناسه، جملهٔ دقیق، سطح شاهد، نسبت با سند/فرضیه، آزمایش و نسخه و run، مسیر `summary.md`، `interpretation.md` و رکورد دقیق `reviews/NNN.json`، hash مدارک اصلی، پروتکل و snapshot را داشته باشد. همچنین این موارد لازم‌اند:

- seedها و تعداد اجرا/نمونه، روش محاسبه و پراکندگی یا عدم قطعیت؛
- منشأ split و تاریخچهٔ استفاده از آن؛ استقلال واقعی از توسعه؛
- baseline و کنترل، بودجهٔ مقایسه و هزینهٔ کامل شامل گذر مقدماتی؛
- runهای منفی/متعارض، موارد حذف و دلیل ازپیش‌ثبت‌شدهٔ حذف؛
- محدودیت، دامنهٔ تعمیم، نمودار/جدول قابل بازتولید و فرمان تولید آن؛
- داوری مرتبط و ADR؛ عبارت مجاز در مقاله و ادعایی که این شاهد پشتیبانی نمی‌کند.

یک run می‌تواند معیار فنی را پاس کند و همچنان برای ادعای علمی کافی نباشد. تبدیل ادعای اکتشافی به تأییدی باید با دادهٔ بازنشده، معیار قفل‌شده و اجرای تازه مستند شود. با اصلاح استراتژی، ادعای قدیمی حذف نشود؛ وضعیت آن و پیوند شاهد جدید افزوده شود.
