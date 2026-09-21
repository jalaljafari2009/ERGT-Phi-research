# Confirm the frozen M2-Q checkpoint on fresh pair-disjoint development data

Experiment: `M2-E003/v003` · stage `M2` · kind `research`.

Hypothesis: The frozen M2-Q v2 checkpoint retains the preregistered auxiliary relation-calibration gates in two raw-text- and pair-disjoint fresh-development windows; Q-only and capacity-matched no-phase controls distinguish proposal access from phase representation without using fresh labels for training or selection.

Goal reference: docs/MATHEMATICAL_SPEC.md sections 5, 14, 15 and 17; docs/SPEC_Q_ANCHOR_ADDENDUM.md; docs/OPERATIONAL_ROADMAP_FA.md operational step 3; ADR-0007

- [Preregistered protocol](protocol.json)
- Notebook: `experiment.ipynb` (generate with the workflow CLI).
- Immutable release lock: `package.json` (created when packaged).
- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.
- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.

An execution success is not a scientific pass. Review gates and limitations before proceeding.

**تغییر v003:** v001 هنگام اجرا و v002 در preflight نشان دادند مسیر lock در بسته نبود. v003 همان lock با hash ثابت را در configs/ قرار می‌دهد؛ هیچ انتخاب علمی تغییر نکرده است.

## طراحی قفل‌شده

این آزمایش checkpoint پذیرفته‌شدهٔ M2-E002 را بدون update روی cohort تازه‌ای از همان
خانواده و بازهٔ ۱ تا ۸ hop می‌سنجد. داده با seed ثبت‌شده تولید شده و هویت ۵۰۴ مثال،
۲۵۲ جفت و دو پنجرهٔ ۲۵۲ مثالی پیش از مشاهدهٔ خروجی مدل در
[قفل توسعهٔ تازه](../../../../configs/m2_e003_fresh_development_lock.json) ثبت شده است.
متن خام و شناسهٔ جفت‌ها با cohort آموزش M0 برخورد ندارند.

سه بازو روی هر دو پنجره گزارش می‌شوند:

1. شبکهٔ phase منجمد با hash ثبت‌شده؛
2. کنترل Q-only برای سنجش اطلاعات proposal بدون هندسهٔ فاز؛
3. طبقه‌بند معمولیِ بدون فاز با ظرفیت تقریباً برابر شبکهٔ phase.

دو کنترل فقط سه epoch ثابت روی fit تاریخی M0 آموزش می‌بینند. برچسب‌های cohort تازه
فقط برای ارزیابی استفاده می‌شوند و حق آموزش، انتخاب checkpoint یا تنظیم آستانه ندارند.
آستانه‌های هر پنجره همان دقت متوازن ۰٫۷۰، recall هر کلاس ۰٫۵۰، کاهش CE پنج درصد نسبت
به حد یکنواخت و معیارهای جدایی offset و عدم فروپاشی phase هستند.

عبور هر دو پنجره تأیید کالیبراسیون توسعه‌ای است. برتری یا شکست نسبت به کنترل‌ها جدا
تفسیر می‌شود و خودکار ادعای علّی phase نمی‌سازد. افق‌های نهایی M8 باز نمی‌شوند، پاسخ
native تغییر نمی‌کند و نتیجهٔ این آزمایش به‌تنهایی M3 را مجاز نمی‌کند.

## ورودی و خروجی

- مرجع M0: `native_ergt_training.pt` با SHA-256 ثبت‌شده در protocol؛
- checkpoint فاز: `selected_m2_q_v2.pt` با SHA-256 برابر
  `8056322cf7c58dd5845f87bf6a89ccbb2ddd76c2d06933a3b5f2fa533688a234`؛
- قرارداد اجرا: [`configs/m2_fresh_v1.json`](../../../../configs/m2_fresh_v1.json)؛
- نتیجه، handoff، ممیزی داده/cache، پیش‌بینی رویدادها و state کنترل‌ها زیر
  `runs/m2_fresh_confirmation/` در bundle خروجی ذخیره می‌شوند.
