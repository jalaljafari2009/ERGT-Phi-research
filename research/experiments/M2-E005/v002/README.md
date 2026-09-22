# Validate an intervention-driven answer and path challenge target with frozen matched readouts

Experiment: `M2-E005/v002` · stage `M2` · kind `research`.

Hypothesis: Preregistered scenario interventions on the frozen native model will create nonzero, finite variation in answer-margin and event/path outcomes in both locked development windows. Capacity-matched readouts trained only on historical M0 fit can then measure how much answer/path sensitivity is present in raw, Q, no-phase and frozen-phase representations without using fresh targets for training or claiming Phase superiority.

Goal reference: docs/MATHEMATICAL_SPEC.md sections 5, 14, 15 and 17; docs/SPEC_Q_ANCHOR_ADDENDUM.md; docs/OPERATIONAL_ROADMAP_FA.md; ADR-0012

- [Preregistered protocol](protocol.json)
- Notebook: `experiment.ipynb` (generate with the workflow CLI).
- Immutable release lock: `package.json` (created when packaged).
- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.
- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.

An execution success is not a scientific pass. Review gates and limitations before proceeding.

## تفاوت دقیق با v001

نسخهٔ v001 پیش از هر محاسبهٔ علمی رد شد، زیرا شمار pairهای hop با cohort قفل‌شدهٔ والد یکی
نبود. این نسخه فقط تعریف پذیرفته‌شده را بازمی‌گرداند: hopهای `1..8` با ۲۸ pair در هر hop و
۲۸ pair unsupported. hash cohort و پنجره‌ها پیش از ارزیابی کنترل می‌شود. هیچ seed، مداخله،
target، بودجهٔ readout، threshold یا سیاست تفسیری تغییر نکرده است.

## طراحی قفل‌شده

مدل native مرجع و شبکهٔ Phase فقط خوانده می‌شوند. هر مثال در حالت کامل و با مداخلهٔ سناریویی
ازپیش‌ثبت‌شده اجرا می‌شود. target اصلی `response_margin_drop` است: حاشیهٔ logit پاسخ درست نسبت
به بهترین پاسخ دیگر در حالت کامل، منهای همان حاشیه زیر مداخله. تغییر صحت پاسخ، زنجیرهٔ event،
recall انتخاب event و پوشش مسیر نیز در سطح مثال ثبت می‌شوند.

چهار scalar readout نزدیک به بودجهٔ ۴۷۰۰ پارامتر (`raw_only`، `q_only`، `no_phase` و
`phase`) دقیقاً ده epoch و فقط روی fit تاریخی M0 آموزش می‌بینند. historical monitor و دو
پنجرهٔ تازه در آموزش، انتخاب، tuning یا تعیین threshold استفاده نمی‌شوند. Q-zero و
Q-permutation تنها در inference اعمال می‌شوند. MAE، RMSE، R²، Pearson و baseline ثابتِ mean
تاریخی گزارش می‌شوند.

## معیار پذیرش و دامنه

هر پنجره باید دست‌کم چهار شکست پاسخ، چهار موفقیت، چهار افت زنجیرهٔ event و انحراف معیار margin
حداقل `1e-6` داشته باشد؛ پاسخ full نیز باید برای همهٔ نمونه‌ها درست و marginها متناهی باشند.
گذر این gateها فقط مناسب‌بودن پنل توسعه‌ای را نشان می‌دهد. رتبه‌بندی readoutها توصیفی است،
ادعای برتری Phase ایجاد نمی‌کند و اجازهٔ M3 نمی‌دهد. M8 نیز بسته می‌ماند.

## نتیجهٔ ثبت‌شده

run محلی package-bound با شناسهٔ `20260922T104000Z-m2e005-v002-local` کامل شد، ولی پنل مرکب
را پاس نکرد. margin پاسخ در هر دو پنجره variation قوی داشت و مداخله ۱۹۲/۲۵۲ پاسخ را خراب کرد؛
در عین حال افت chain و path در هر دو پنجره صفر بود. [تفسیر](runs/20260922T104000Z-m2e005-v002-local/interpretation.md)
و [ADR-0014](../../../decisions/ADR-0014.md) نتیجه را `revise` ثبت می‌کنند. readout Phase از نظر
توصیفی بهتر بود، اما Q-zero/permute آن هم‌اندازه یا بهتر از full بود؛ بنابراین ادعای برتری Phase
ایجاد نشد. مرحلهٔ بعد مداخلهٔ پیش‌حل event با cohort توسعه‌ای تازه است و M3/M8 بسته‌اند.
