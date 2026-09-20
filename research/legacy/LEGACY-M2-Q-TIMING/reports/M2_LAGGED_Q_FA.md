# M2 — قرارداد Q تأخیری و جداشده

## کاری که تکمیل شد

خروجی سه‌گانه‌ی `ProbeNative` اکنون یک آداپتر تایپ‌شده دارد:

- `event_source_logits` برای توزیع مبدأ هر slot؛
- `event_target_logits` برای توزیع مقصد؛
- `event_relation_logits` برای چهار حالت رابطه، با کلاس null در ستون صفر.

تابع `native_proposal_q` فقط همین proposalهای بومی و ماسک padding را می‌گیرد و
برای یال‌های کاندید، ماتریس sparse زیر را می‌سازد:

```text
Q[i,j,r] = sum_u valid_slot[u] * Psrc[u,i] * Prel[u,r] * Ptgt[u,j]
```

جرم کلاس null حذف نمی‌شود؛ جرم غیر-null همان حضور رویداد است و فقط یک‌بار در
`Q` مصرف می‌شود. سپس `q_e=sum_r Q[e,r]` و `pi_e=Q[e,:]/q_e` هر دو detached
می‌شوند تا ورودی حل‌گر phase از مسیر گرادیان native جدا بماند.

## زمان‌بندی

کلاس `LaggedQ` با مقدار اولیه‌ی `Q^0=empty`، مصرف نسخه‌ی قبلی با `consume` و
ثبت نسخه‌ی جدید فقط پس از پایان outer step با `commit` این قرارداد را enforce
می‌کند. ثبت Q دارای گرادیان یا مقدار غیرمتناهی رد می‌شود و `consume` یک clone
برمی‌گرداند تا تغییر بیرونی، حافظه‌ی lagged را دست‌کاری نکند.

## کنترل‌ها

تست‌ها شکل sparse، ماسک padding، حذف کلاس null، نرمال‌سازی `pi`، یال self/خارج
از دامنه، جداسازی gradient و ایزوله‌بودن حافظه‌ی lagged را بررسی می‌کنند.

این تغییر هنوز phase را به geometry وصل نمی‌کند و M3 را مجاز نمی‌کند. گام بعدی
یک shadow calibration با Q واقعی و سپس آزمون zero-effect است؛ تا عبور هر دو،
`M3_authorized_by_results` باید `false` بماند.
