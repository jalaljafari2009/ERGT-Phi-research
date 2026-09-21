# اعتبارسنجی قرارداد M2-Q v2 — ۲۱ سپتامبر ۲۰۲۶

این گزارش فقط تحویل **گام ۱ برنامهٔ عملیاتی** را ثبت می‌کند: تثبیت قرارداد
Q-conditioned anchor. در این گام آموزش، اجرای Colab، انتخاب checkpoint یا فعال‌سازی
M3 انجام نشده است.

## اقلام تحویل‌شده

- افزونهٔ معماری: [`docs/SPEC_Q_ANCHOR_ADDENDUM.md`](../../docs/SPEC_Q_ANCHOR_ADDENDUM.md)
- پیکربندی ماشین‌خوان: [`configs/m2_q_v2.json`](../../configs/m2_q_v2.json)
- تصمیم معماری: [ADR-0006](../decisions/ADR-0006.md)
- validator: [`ergt_phi/m2_q_contract.py`](../../ergt_phi/m2_q_contract.py) و
  [`scripts/check_m2_q_contract.py`](../../scripts/check_m2_q_contract.py)

## کنترل‌های انجام‌شده

| کنترل | نتیجه |
|---|---|
| `python -B -X utf8 scripts/check_m2_q_contract.py` | پاس؛ schema برابر `ergt-phi-m2-q-anchor-config-v1` و revision برابر `m2-q-v2` |
| هش config | `ad68014af044da8b6ed163db80a83bea8aabfe9d5a466a652142746661c9e1d4` |
| هش افزونه | `d93b4013d20f5524eedc4954a782fa02f4d869fe0e0b836c84fb193ddf993f68` |
| `python -B -X utf8 scripts/check_spec_lock.py --require-read-only` | پاس؛ ۲۳۴۵۰ بایت، SHA-256 برابر `04ac57cc76395f7b032356bcac38b70b796233ac4aa59ba26a6b2474977c5d90` و ReadOnly فعال |
| parse کردن `configs/m2_q_v2.json` و `research/goals.json` | پاس |
| `pytest tests/test_m2_q_contract.py -q` | ۴ تست پاس؛ قرارداد معتبر و رد drift در bypass، حافظهٔ Q و خروجی native |
| `pytest tests/test_colab_workflow.py -q` با basetemp محلی | ۸ تست پاس |

اجرای نخست pytest پیش از ورود به تست‌ها به‌علت ACL پوشهٔ موقت sandbox متوقف شد.
بازاجرا با همان محیط پایتون و دسترسی عادی ویندوز، با `basetemp` داخل workspace، هر
۸ تست workflow را پاس کرد. همراه ۴ تست قرارداد، مجموع کنترل‌های pytest این milestone
۱۲ تست است. پوشه‌های موقت پس از آزمون حذف شدند.

## جمع‌بندی دامنه

قرارداد اکنون فرمول anchor، ترتیب پیش‌گذر، جدایی `Q_pre` و `Q_prev`، محدودهٔ freeze،
bypass و حساب هزینه را بدون ابهام نسخه‌بندی می‌کند. مرز runner آفلاینِ shadow با
public/integrated forward نیز صریح است تا کالیبراسیون coupling صفر شرط zero-effect را
نقض نکند. وضعیت عمداً
`contract_frozen_implementation_pending` است. گام بعدی ساخت runner منطبق، انتخاب
اولین checkpoint واجد شرایط، freeze فوری و آزمون reload/resume در فرایند تازه است.
