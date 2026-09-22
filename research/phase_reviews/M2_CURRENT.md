# M2 — وضعیت شروع workflow نسخه‌دار

- وضعیت: handoff مهندسی گام ۲ پذیرفته؛ تأیید توسعهٔ تازه ناقص و ورود به M3 هنوز مجاز نشده است.
- هدف: `GOAL-001` در `research/goals.json`.
- شواهد: `research/legacy/index.json`، رکورد `LEGACY-M2-Q-CALIBRATION`.
- برنامهٔ مبنا: `docs/OPERATIONAL_ROADMAP_FA.md`، گام‌های ۱ تا ۵.

کالیبراسیون Q-conditioned عدد ۸۱٫۲۱۶٪ دقت متوازن پایش و ۳۲٫۱۸۷٪ کنترل برچسب تصادفی
را ثبت کرده است. monitor از cohort آموزش مدل مرجع است؛ شاهد تأییدی مستقل نیست.
checkpoint منتخب phase اکنون ذخیره شده و freeze/reload/resume آن در run واردشده تأیید شده است.
Q-shadow از مدل جدا بوده؛ نسبت‌دادن آن به قبولی حلقهٔ lagged یا اتصال واقعی صحیح نیست.

کارهای لازم قبل از phase review بعدی:

1. **انجام شد — ۲۱ سپتامبر ۲۰۲۶:** افزونهٔ معماری anchor، زمان‌بندی گذر مقدماتی،
   config ماشین‌خوان و validator در `ADR-0006` ثبت شدند؛ این تکمیل قرارداد است، نه run.
2. **پذیرفته شد — ۲۲ سپتامبر ۲۰۲۶:** run واردشدهٔ `M2-E002/v001`
   checkpoint کامل را در epoch ۳/step ۷۲ freeze کرد و هر ۱۹ gate را پاس کرد.
3. **پذیرفته شد در دامنهٔ مهندسی:** reload metric و resume کامل در فرایند تازه، منع
   update پس از freeze و parity خروجی native پاس شدند؛ [ADR-0007](../decisions/ADR-0007.md).
4. **پاس شد با محدودیت — ۲۲ سپتامبر ۲۰۲۶:** `M2-E003/v004` checkpoint منجمد
   را بدون تنظیم مجدد روی دو پنجرهٔ توسعهٔ تازه، pair/text-disjoint و در برابر Q-only
   و no-phase هم‌ظرفیت می‌سنجد. v001 پیش از محاسبه شکست خورد و v002 در preflight
   بستهٔ ناقص داشت؛ v003 پیش از metric خطای حساب ظرفیت را آشکار کرد. v004 عرض درست ۳۵
   را با تست صریح دارد. phase در دو پنجره `0.7800/0.7712` و هر دو کنترل `1.0/1.0`
   balanced accuracy داشتند؛ کالیبراسیون تعمیم یافت ولی سود اختصاصی phase دیده نشد.
5. **پاس شد در دامنهٔ ممیزی — ۲۲ سپتامبر ۲۰۲۶:** `M2-E004/v001` با checkpoint
   ثابت Q-zero، Q-permutation و raw-only را اجرا کرد. Q-only از `1.0` به `0.333/0`
   افت کرد و raw-only نزدیک شانس بود. relation در جفت‌های دارای answer flip ثابت بود؛
   پنل native answer/path روی همین cohort سقف کامل داشت؛ [ADR-0012](../decisions/ADR-0012.md).
6. **اجرا شد و نیازمند بازنگری است — ۲۲ سپتامبر ۲۰۲۶:** `M2-E005/v002` target پیوستهٔ
   response-margin و پنل intervention را روی همان cohort اجرا کرد. در هر پنجره ۱۹۲ پاسخ خراب و
   ۶۰ پاسخ درست باقی ماند و margin متمایز بود، اما event-chain و path degradation هر دو صفر
   بودند. Phase-full در readout توصیفی بهتر بود، ولی Phase-zero/permute هم‌اندازه یا بهتر شد؛
   بنابراین شاهد اختصاصی Phase ایجاد نشد؛ [ADR-0014](../decisions/ADR-0014.md).
7. **اقدام بعد:** پیش از اصلاح آداپتر governing، M2-E006 باید intervention را به سطح
   proposal/event و پیش از solver منتقل کند و یک sham هم‌بودجه داشته باشد. ابتدا اثر مکانیزمی
   آن بر selected-event recall، chain exactness یا path coverage ثابت شود؛ ارزیابی تأییدی روی
   cohort توسعه‌ای تازه و disjoint انجام شود. پنجره‌های قبلی دیگر تازه محسوب نمی‌شوند. relation
   accuracy فقط metric کمکی است و M3/M8 بسته‌اند.

این فایل مرور انتقالی تاریخچه است. تصمیم بعدی باید فایل phase review تازه با پیوند به
runهای نسخه‌دار و ADR بسازد. `M2_complete=true` در manifest قدیمی معنای قبولی عددی
همان آزمایش را دارد؛ آن فایل برای پنهان‌کردن این محدودیت بازنویسی نمی‌شود.
