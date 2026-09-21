# افزونهٔ معماری Q-conditioned anchor — نسخهٔ M2-Q v2

- تاریخ تثبیت قرارداد: ۲۱ سپتامبر ۲۰۲۶.
- وضعیت: **قرارداد طراحی پذیرفته‌شده؛ پیاده‌سازی runner و اجرای علمی هنوز لازم است.**
- تصمیم: [ADR-0006](../research/decisions/ADR-0006.md).
- پیکربندی ماشین‌خوان: [`configs/m2_q_v2.json`](../configs/m2_q_v2.json).
- مبنا: [`MATHEMATICAL_SPEC.md`](MATHEMATICAL_SPEC.md)، با SHA-256 ثبت‌شده در
  [`research/specification.lock.json`](../research/specification.lock.json).

این سند یک افزونهٔ نسخه‌دار برای مسیر A است. متن سند ریاضی پایه را تغییر نمی‌دهد و
هیچ نتیجهٔ علمی تازه یا مجوز ورود به M3 ایجاد نمی‌کند.

## ۱. مسئله و تفاوت دقیق با مبنا

بخش ۵ سند پایه anchor را فقط از `Psi0` می‌سازد:

```text
anchor_iw = pi * tanh(MLP(LN(Psi0_i)))
```

در شواهد تاریخی M2، این ورودی اطلاعات رابطه را به اندازهٔ کافی منتقل نکرد. شاخهٔ
M2-Q با افزودن خلاصهٔ جداشدهٔ پیشنهادهای بومی به `Psi0` نتیجهٔ توسعهٔ امیدوارکننده‌ای
ثبت کرد، اما checkpoint منتخب را تحویل نداد. نسخهٔ v2 قرارداد همان ایده را دقیق می‌کند:

```text
raw tokens + mask
        ↓
reference prepass: frozen/eval, independent object
        ↓
completed native field snapshot → ProbeNative, before answer solving
        ↓
detached Q_pre → node source/target marginals
        ↓
x_i = concat(Psi0_i, log1p(source_Q_i), log1p(target_Q_i))
        ↓
anchor = pi * tanh(MLP(LN(x)))       [built once and fixed]
```

پس تفاوت نسخهٔ v2 با بخش ۵ فقط **ورودی شبکهٔ anchor** و هزینهٔ پیش‌گذر لازم برای
ساخت آن است. عملگرهای native، ورودی عمومی مدل، حل‌گر پاسخ و قراردادهای بخش‌های دیگر
سند پایه با این تصمیم تغییر نمی‌کنند.

## ۲. قرارداد پیش‌گذر مرجع

1. ورودی مجاز فقط `raw_token_ids` و `attention_mask` همان نمونه است. برچسب، gold edge،
   جواب، شاهد مسیر و خروجی حل‌گر پاسخ وارد پیش‌گذر نمی‌شوند.
2. یک شیء مرجع مستقل از مدل فعال ساخته می‌شود، checkpoint پذیرفته‌شدهٔ M0 را با
   `strict=True` بارگذاری می‌کند، در حالت `eval` می‌ماند و همهٔ پارامترهایش frozen هستند.
3. مرجع از ورودی خام، `Psi0` و میدان native کامل را می‌سازد. روی snapshot کامل‌شده،
   `ProbeNative` اجرا می‌شود و پیش از candidate answer solving متوقف می‌شود.
4. حداکثر یک پیش‌گذر و یک probe برای هر ورودی/forward مجاز است. بازاستفاده از cache
   فقط با برابری hash ورودی خام، mask، checkpoint مرجع و نسخهٔ config مجاز است.
5. tensorهای proposal پیش از ورود به شاخهٔ phase جدا می‌شوند. مرجع نه gradient می‌گیرد،
   نه optimizer دارد و نه RNG مدل فعال را مصرف می‌کند.
6. اگر در آینده وزن‌های native مسیر فعال باز شوند، شیء مرجع همچنان مستقل، frozen و
   بارگذاری‌شده از checkpoint ثبت‌شده باقی می‌ماند.

## ۳. ساخت Q مقدماتی و anchor

برای slot معتبر `u` و رابطهٔ غیر-null `r`، Q مطابق بخش ۴ سند پایه ساخته می‌شود:

```text
Q_pre[i,j,r] = sum_u Psrc[u,i] * Prel[u,r] * Ptgt[u,j]
```

جرم event-presence که در `Prel` وجود دارد دوباره ضرب نمی‌شود. padding و self-edge
حذف می‌شوند. سپس برای هر node و relation:

```text
source_Q[i,r] = sum_j Q_pre[i,j,r]
target_Q[i,r] = sum_j Q_pre[j,i,r]
x_i = concat(Psi0_i, log1p(source_Q[i,:]), log1p(target_Q[i,:]))
anchor_iw = pi * tanh([W2 SiLU(W1 LN(x_i))]_w)
```

`log1p` صفر را صفر نگه می‌دارد و جرم‌های بزرگ را متناهی می‌کند. anchor برای یک forward
فقط یک‌بار ساخته می‌شود و در همهٔ outer stepها و inner solveهای همان forward ثابت است.
بازسازی anchor از Q تازهٔ هر گام، معماری دیگری است و در v2 ممنوع است.

در M2، gold endpoints و gold relation فقط **بعد از cacheشدن ویژگی‌های raw-only** برای
indexing و loss کالیبراسیون مجازند. آن‌ها candidate set یا ورودی anchor نیستند.

## ۴. جدایی Q مقدماتی از Q تأخیری

دو Q نقش متفاوت دارند و نباید جای یکدیگر مصرف شوند:

| نام | منشأ | نقش | زمان مصرف |
|---|---|---|---|
| `Q_pre` | پیش‌گذر مرجع مستقل و frozen | فقط ساخت anchor ثابت | پیش از آغاز مسیر فعال |
| `Q_prev` | `ProbeNative` مسیر فعال پس از گام کامل‌شده | ساخت patch و mixture حل phase | فقط در گام بعد |

مسیر فعال با state تازه و `Q_prev=empty` آغاز می‌شود. `Q_pre` به حافظهٔ lagged نوشته
نمی‌شود. بنابراین outer step اول native است و proposal گام `t` فقط می‌تواند روی گام
`t+1` اثر بگذارد. این جدایی باید با شناسهٔ snapshot/step در M4 آزموده شود.

## ۵. حالت خاموش و coupling صفر

شرط bypass قبل از ساخت شیء مرجع، cache، پیش‌گذر، probe یا anchor بررسی می‌شود:

```text
if phase_mode == "disabled" or coupling == 0:
    return original_native_forward(raw_token_ids, attention_mask)
```

در bypass نباید فراخوانی اضافه، تخصیص RNG، تغییر حالت ماژول، cache یا هزینهٔ پیش‌گذر
وجود داشته باشد. برابری خروجی، تصمیم سخت، gradient، یک optimizer step و RNG با baseline
در دروازهٔ zero-effect پیش از M3 آزموده می‌شود. این افزونه به‌تنهایی آن دروازه را پاس نمی‌کند.

این bypass قرارداد `public_forward` و مسیر فعال یکپارچه است. runner کالیبراسیون M2
یک entrypoint پژوهشیِ آفلاین و صریح است: حتی با coupling صفر می‌تواند ویژگی‌های shadow
پیش‌گذر را مشاهده و آموزش کمکی دهد، اما خروجی native را جایگزین یا اصلاح نمی‌کند و
به‌عنوان اجرای public forward گزارش نمی‌شود. به این ترتیب اندازه‌گیری M2 با شرط
zero-effect مسیر قابل‌استفادهٔ مدل مخلوط نمی‌شود.

## ۶. دامنهٔ M2 و پارامترهای قابل‌آموزش

M2-Q v2 یک آزمایش shadow است:

- entrypoint آن runner آفلاین کالیبراسیون است، نه `public_forward` مدل؛
- native geometry و answer execution تغییر نمی‌کنند؛ coupling برابر صفر است؛
- مرجع frozen است و فقط `phase.network` و `phase.offset_raw` قابل‌آموزش‌اند؛
- offsetها پس از انتخاب checkpoint freeze می‌شوند؛
- loss رابطه فقط روی دادهٔ training/development ثبت‌شده محاسبه می‌شود؛
- نتیجهٔ کالیبراسیون، ادعای بهبود پاسخ نهایی یا پذیرش M3 نیست.

آستانه‌های موجود حفظ می‌شوند: balanced accuracy حداقل ۰٫۷۰، recall هر کلاس حداقل
۰٫۵۰، کاهش CE حداقل ۵٪، جدایی offset حداقل ۱ رادیان و phase resultant حداکثر ۰٫۹۸،
در دو پنجرهٔ متوالی. اولین checkpoint واجد دو پنجره باید همان لحظه selected و frozen
شود؛ ادامهٔ آموزش حق جایگزینی آن با وزن نامعتبر بعدی را ندارد.

## ۷. checkpoint، بازیابی و بودجه

اجرای بعدی باید `initial`، `latest`، `selected` یا `rejected` را همراه مدل، optimizer،
scheduler/scaler در صورت وجود، RNG، data cursor، وضعیت freeze، contract hash، source/data
hash و گروه‌های قابل‌آموزش ذخیره کند. `reload` در فرایند تازه باید metric منتخب را
بازتولید کند و `resume` باید از step boundary ثبت‌شده ادامه دهد.

بودجهٔ معماری v2 شامل یک پیش‌گذر کامل native و یک proposal probe برای هر ورودی است؛
این هزینه همراه peak memory، cache construction و آموزش phase گزارش می‌شود. اجرای M2
فعلی CPU، `float32`، یک thread، ۲۰ epoch و batch size ۱۶ است. سقف زمان دیواری و تعداد
اجرای تأییدی باید در protocol آزمایش، پیش از ساخت نوت‌بوک، مقدار عددی بگیرند.

## ۸. معیار تکمیل این گام و کار باقی‌مانده

این گام وقتی کامل است که این افزونه، config ماشین‌خوان، validator، ADR و پیوندهای وضعیت
با هم سازگار باشند و قفل سند پایه برقرار بماند. تکمیل این قرارداد به معنی وجود runner
v2، checkpoint یا نتیجهٔ تازه نیست.

گام بعدی مسیر A اصلاح runner بر اساس همین قرارداد است: انتخاب فوری checkpoint، freeze،
load/resume و کنترل عدم تغییر مرجع. سپس protocol و notebook نسخه‌دار ساخته و نتیجهٔ تازه
وارد workflow می‌شود. آداپتر native، zero-effect و M3 همچنان پس از تحویل معتبر M2 هستند.
