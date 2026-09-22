# تفسیر پژوهشی اجرای 20260922T072000Z-m2e004-v001-local

- آزمایش: `M2-E004/v001`
- source commit: `e34f7448d9d3fa3f44a822640dd0bca0ca81d4df`
- package SHA-256: `e452e916fb19e5c5720d718760fb7af54bd43db97d6880111d883a29523d4c5f`
- bundle SHA-256: `f961ea44cdd74224606c1f33000158a3d82ccd83fc39c2f4035b88858b86fd6e`
- مدارک: [summary](summary.md)، [evaluation](evaluation.json)، [run](run.json) و [artifacts](artifacts.json)

## صحت اجرا

هر ۲۰ gate پاس شد. baselineهای کامل M2-E003/v004 با دقت ممیز شناور ثبت‌شده
بازتولید شدند. cohort تازه، دو پنجره و جدایی آن‌ها با lock قبلی یکسان بودند؛ هیچ
برچسب تازه‌ای وارد آموزش یا انتخاب نشد. checkpoint مرجع، phase، Q-only و no-phase
و فایل‌های ورودی پیش و پس از اجرا hash/state یکسان داشتند. M3 و M8 باز نشدند.

## نتیجهٔ ablation

| بازو | پنجره ۱: کامل / Q صفر / Q permute | پنجره ۲: کامل / Q صفر / Q permute |
|---|---:|---:|
| Q-only | 1.000 / 0.333 / 0.000 | 1.000 / 0.333 / 0.000 |
| no-phase | 1.000 / 0.322 / 0.000 | 1.000 / 0.308 / 0.000 |
| phase | 0.780 / 0.333 / 0.117 | 0.771 / 0.333 / 0.108 |

اعداد balanced accuracy هستند. در Q-only، permutation پیش‌بینی همهٔ eventها را عوض
کرد و صفرکردن Q حدود ۶۳٪ و ۶۶٪ پیش‌بینی‌ها را در دو پنجره تغییر داد. phase نیز با
صفرکردن Q حدود ۰٫۴۴ دقت متوازن از دست داد و permutation حدود ۷۴–۷۵٪ پیش‌بینی‌هایش
را عوض کرد. پس هر سه readout در موفقیت کامل خود به هویت معنایی کانال‌های Q وابسته‌اند.

raw-only با ۴۷۵۱ پارامتر در برابر ۴۷۰۰ پارامتر phase، دقت متوازن ۰٫۳۳۳ و ۰٫۳۲۷
داشت. بنابراین relation از `Psi0` دو endpoint به‌سادگی خوانده نشد. این نکته مهم است:
یافتهٔ shortcut به معنی تزریق gold label یا یک shortcut مستقیم در endpoint خام نیست؛
مسیر توضیح‌دهنده، پردازش متن کامل توسط native proposal و فشرده‌شدن relation در Q است.

## ممیزی target

در هر ۴۵۱۶ event، relation surface دقیقاً در event anchor متن خام حضور داشت. relation
sequence در هر ۲۵۲ جفت counterfactual یکسان بود، ولی در هر ۲۲۴ جفت supported پاسخ درست
عوض شد. بنابراین target رابطه برای کالیبراسیون محلی معتبر است، اما به‌تنهایی نمی‌تواند
انتخاب میان دو پاسخ counterfactual را تفکیک کند. نتیجهٔ کامل Q-only روی این target
شاهد کیفیت پاسخ یا مزیت phase نیست.

پنل نزدیک‌تر به هدف، مدل native منجمد را روی answer accuracy، counterfactual pair
exact، program-event recall/precision، event-chain exact، selected-event recall و
path coverage سنجید. همهٔ این metricها در هر دو پنجره ۱٫۰ بودند. این نتیجه کیفیت
مرجع را در این cohort نشان می‌دهد، اما یک سقف کامل است: هیچ خطای پاسخ یا مسیر برای
یادگیری، شرطی‌سازی یا مقایسهٔ یک آداپتر phase وجود ندارد.

## تصمیم و محدودیت

ممیزی در دامنهٔ ثبت‌شده `pass` است. ریسک shortcut/عدم‌هم‌راستایی target relation
پشتیبانی شد، ولی raw-only آن را توضیح نداد و هیچ ادعای data leakage ثبت نمی‌شود.
target پاسخ/مسیر از نظر علمی مناسب‌تر است، اما همین cohort به‌علت سقف کامل برای آزمایش
سود phase کافی نیست.

گام بعد باید هنوز در M2 بماند: یک پنل توسعهٔ چالش‌دار و قفل‌شده با margin انتخاب
candidate، event-chain/path loss و حساسیت به مداخلات مجاز تعریف شود تا variation
غیرصفر ایجاد کند. دادهٔ M8 باز نمی‌شود و تا وجود target تمایزبخش، کنترل‌های متناظر و
قرارداد عدم leakage، M3 مجاز نیست.

## مادهٔ مقاله

عبارت مجاز: «در ممیزی توسعه‌ای frozen، صفر یا permutation کانال‌های relation در Q
دقت readoutهای relation را تا شانس یا پایین‌تر کاهش داد؛ raw endpoint-only نزدیک شانس
بود. relation sequence در جفت‌هایی که پاسخ عوض می‌شد ثابت ماند، بنابراین relation
کالیبراسیون target مناسبی برای ادعای کیفیت انتخاب پاسخ نبود. پنل native همان cohort
سقف کامل داشت و برای مقایسهٔ phase تمایزبخش نبود.»

