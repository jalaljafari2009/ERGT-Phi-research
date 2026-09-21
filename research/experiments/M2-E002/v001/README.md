# M2-E002/v001 — تحویل checkpoint واقعی M2-Q v2

این آزمایش گام ۲ برنامهٔ عملیاتی را اجرا می‌کند: اولین state که در دو پنجرهٔ متوالی
تمام معیارهای کالیبراسیون را پاس کند، همان لحظه ذخیره و frozen می‌شود. سپس metric آن
در یک فرایند تازه بازتولید و برابری ادامهٔ آموزش با اجرای پیوسته آزموده می‌شود.

## هویت ثابت

- مرحله: `M2`، مسیر A، نوع `research`
- قرارداد معماری: [`configs/m2_q_v2.json`](../../../../configs/m2_q_v2.json)
- تصمیم: [ADR-0006](../../../decisions/ADR-0006.md)
- [پروتکل پیش‌ثبت‌شده](protocol.json)
- [نوت‌بوک Colab](experiment.ipynb) و [هش آن](notebook.sha256)
- ورودی بزرگ: checkpoint مرجع M0 با SHA-256 برابر
  `82d1220ee4bcc7f721e8159ea3515fe5992a92a184a611db6223e292a14eb065`

## چه چیزی آزموده می‌شود؟

- شش معیار readiness با آستانه‌های تثبیت‌شده و دو پنجرهٔ متوالی؛
- انتخاب فوری اولین checkpoint واجد شرایط و جلوگیری از update بعد از freeze؛
- reload در Python تازه و برابری دقیق metric؛
- resume در Python تازه و برابری کامل مدل، optimizer، RNG، cursor و modeها؛
- عدم تغییر وزن، gradient، خروجی و تصمیم سخت مدل مرجع؛
- کنترل deterministic برچسب درهم‌ریخته؛
- زمان کامل پیش‌گذر و probe به‌عنوان بخشی از هزینه.

۱۹ gate و ۱۲ artifact اجباری در `protocol.json` ثبت شده‌اند. نبود checkpoint منتخب
یا هر artifact الزامی، اجرای ناموفق را به نتیجهٔ قابل‌قبول تبدیل نمی‌کند.

## اجرای محلی پیش از انتشار

اجرای مهندسی مستقیم روی ۵۰۴ نمونه، با همان command پروتکل، runner نهایی را بررسی کرد:
checkpoint نخست در epoch ۳ و step ۷۲ انتخاب شد، balanced accuracy پایش `0.7555267`
و کنترل shuffled برابر `0.3218692` بود. همهٔ ۱۹ gate محلی پاس شدند. این خروجی در
`runs/` فقط scratch اعتبارسنجی است؛ run واردشدهٔ Colab یا تأیید توسعهٔ تازه محسوب نمی‌شود.

## اجرای Colab و بازگشت نتیجه

release قفل‌شده از commit `7891db73a05ef2d7b7fcbe4f33f94235b3ca2366` ساخته شد:

- ZIP: `research/packages/M2-E002-v001-59d6406911ac-source.zip`
- SHA-256 بسته: `8bc4ef850b4fe02844d93aeea48b83fa44a966a4fe9537390bd202f831864ed7`
- lock قابل حمل: [package.json](package.json)
- موجودی ۱۷۰ فایل: [source_manifest.json](source_manifest.json)
- پوشهٔ آمادهٔ انتخاب فایل در این checkout:
  `research/packages/M2-E002-v001-upload/`

در Colab، `experiment.ipynb` را Run all کنید. ابتدا ZIP سورس و `package.json` همین
نسخه و سپس فایل checkpoint مرجع M0 را بارگذاری کنید. نوت‌بوک hash هر سه ورودی را
پیش از اجرا کنترل می‌کند. ZIP نتیجه را بدون تغییر در `research/inbox/` قرار دهید.

نتیجه پس از import در `runs/<run_id>/` همین نسخه ثبت می‌شود و سپس review مستقل
`pass`، `revise` یا `inconclusive` می‌گیرد. موفقیت این آزمایش فقط تحویل checkpoint
M2 را بررسی می‌کند؛ تأیید دادهٔ تازه گام ۳ است و M3 همچنان مجاز نیست.

## نتیجهٔ ثبت‌شده

release قفل‌شده در اجرای محلی package-bound با شناسهٔ
`20260921T210953Z-48abf488-local` اجرا و bundle آن وارد شد. هر ۱۹ gate پاس شدند و
checkpoint منتخب با SHA-256 برابر
`8056322cf7c58dd5845f87bf6a89ccbb2ddd76c2d06933a3b5f2fa533688a234`
ثبت شد. [خلاصه](runs/20260921T210953Z-48abf488-local/summary.md)،
[تفسیر](runs/20260921T210953Z-48abf488-local/interpretation.md) و
[review](runs/20260921T210953Z-48abf488-local/reviews/001.json) دامنهٔ نتیجه را مشخص
می‌کنند. [ADR-0007](../../../decisions/ADR-0007.md) handoff گام ۲ را پذیرفت و گام ۳
را به تأیید توسعهٔ تازه سپرد؛ M3 همچنان بسته است.
