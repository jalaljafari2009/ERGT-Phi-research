# M2 — کالیبراسیون shadow با Q واقعی

## چرا پروتکل عوض شد؟

کالیبراسیون قبلی از anchor مستقل یا context نهایی می‌خواست رابطه را از دو
endpoint بازسازی کند و به حد پذیرش نمی‌رسید. آزمون جداگانه نشان داد که خود
proposal بومی رابطه را با دقت متوازن ۱۰۰٪ روی پایش حمل می‌کند. بنابراین یک
پروتکل صریح و نسخه‌دار تعریف شد: context رابطه از Q بومیِ detached می‌آید و
`Psi0` فقط برای مختصات phase باقی می‌ماند.

این یک اصلاح پروتکل پژوهشی است و نتیجهٔ M2 قبلی را retroactively تغییر نمی‌دهد.

## روش

- همان cohort آموزش M0 و همان split جفتی استفاده شد: ۳۸۲ fit و ۱۲۲ monitor؛
- برای هر raw input، snapshot بومی و `ProbeNative` بدون label و بدون answer head
  اجرا شد؛
- Q روی همهٔ جفت‌های معتبر ساخته شد و حاشیهٔ source/target آن به `Psi0`
  اضافه شد؛
- Q، context و وزن‌های native detached و frozen بودند؛
- gold endpoint فقط بعد از cache شدن featureها برای loss relation استفاده شد؛
- آموزش shadow با ۲۰ epoch و کنترل جداگانهٔ برچسب‌های درهم‌ریخته انجام شد.

## نتیجه

| معیار | واقعی | برچسب درهم‌ریخته |
|---|---:|---:|
| دقت متوازن پایش | **۸۱٫۲۲٪** | ۳۲٫۱۹٪ |
| دقت معمول پایش | ۸۱٫۲۰٪ | ۳۱٫۲۲٪ |

هر سه recall رابطه در پایش بالاتر از ۵۰٪ بود، CE از مقدار اولیه کاهش یافت،
offsetها از هم جدا ماندند و phase collapse رخ نداد. بنابراین این **پروتکل revised
M2 قبول شد**.

## مرز نتیجه

این موفقیت فقط کالیبراسیون shadow با context Q است. geometry native هنوز فعال
نشده، answer head وارد loss نشده و parity وزن/خروجی native حفظ شده است. به همین
دلیل `M3_authorized_by_results=false` باقی می‌ماند تا آزمون مستقل M3 با coupling
ضعیف، `nu=0`، mask و field ثابت اجرا و بررسی شود.

نتیجهٔ کامل در [m2_q_calibration.json](../manifests/m2_q_calibration.json) و
پروتکل قبل از مشاهدهٔ نتیجه در `runs/m2_q_calibration/protocol.json` ثبت شده است.
