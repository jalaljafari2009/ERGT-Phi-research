# Validate an intervention-driven answer and path challenge target with frozen matched readouts

Experiment: `M2-E005/v001` · stage `M2` · kind `research`.

Hypothesis: Preregistered scenario interventions on the frozen native model will create nonzero, finite variation in answer-margin and event/path outcomes in both locked development windows. Capacity-matched readouts trained only on historical M0 fit can then measure how much answer/path sensitivity is present in raw, Q, no-phase and frozen-phase representations without using fresh targets for training or claiming Phase superiority.

Goal reference: docs/MATHEMATICAL_SPEC.md sections 5, 14, 15 and 17; docs/SPEC_Q_ANCHOR_ADDENDUM.md; docs/OPERATIONAL_ROADMAP_FA.md; ADR-0012

- [Preregistered protocol](protocol.json)
- Notebook: `experiment.ipynb` (generate with the workflow CLI).
- Immutable release lock: `package.json` (created when packaged).
- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.
- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.

An execution success is not a scientific pass. Review gates and limitations before proceeding.

## طراحی قفل‌شده

این آزمایش همان cohort توسعه‌ای و دو پنجرهٔ pair-disjoint پذیرفته‌شده در
`M2-E003/v004` را بازتولید می‌کند. مدل native مرجع و شبکهٔ Phase انتخاب‌شده فقط خوانده
می‌شوند و hash و state آن‌ها پیش و پس از اجرا کنترل می‌شود. M8 باز نمی‌شود و هیچ نمونه‌ای
از افق نهایی وارد این آزمایش نیست.

برای هر مثال، مدل native یک بار در حالت کامل و یک بار با مداخلهٔ متناسب با سناریو اجرا
می‌شود: حذف action، cone، boundary، transport، terminal mass، memory geometry، یا
multiscale backbone. target اصلی برابر است با حاشیهٔ logit پاسخ درست نسبت به بهترین پاسخ
دیگر در حالت کامل، منهای همان حاشیه زیر مداخله. علامت مثبت یعنی مداخله تصمیم frozen native
را ضعیف کرده است. تغییر صحت پاسخ، افت exactness زنجیرهٔ event، افت recall انتخاب event و
افت پوشش مسیر نیز برای هر مثال ثبت می‌شوند.

چهار scalar readout با بودجهٔ نزدیک به ۴۷۰۰ پارامتر ساخته می‌شوند: `raw_only`، `q_only`،
`no_phase` و `phase`. همه دقیقاً ده epoch و فقط روی fit تاریخی M0 آموزش می‌بینند. historical
monitor و هر دو پنجرهٔ تازه از آموزش، انتخاب، tuning و تعیین threshold کنار گذاشته شده‌اند.
روی دادهٔ تازه، Q-zero و Q-permutation صرفاً در inference برای سه خانوادهٔ وابسته به Q
اجرا می‌شوند. metricها MAE، RMSE، R² و Pearson در برابر target پیوسته‌اند و خطای predictor
ثابتِ mean تاریخی نیز گزارش می‌شود.

## معیار پذیرش پنل

در هر پنجره باید پاسخ کامل برای همهٔ نمونه‌ها درست باشد، مداخله دست‌کم چهار شکست و چهار
موفقیت باقی‌مانده ایجاد کند، دست‌کم چهار زنجیرهٔ event افت کند، و انحراف معیار افت margin
حداقل `1e-6` و متناهی باشد. این thresholdها بعد از مشاهدهٔ نتیجه تغییر نمی‌کنند. گذر پنل
فقط نشان می‌دهد target توسعه‌ای برای diagnostic بعدی support دارد؛ هیچ رتبه‌بندی readout
در این نسخه gate نیست و هیچ نتیجه‌ای ادعای برتری Phase یا اجازهٔ M3 ایجاد نمی‌کند.

## artifactهای قابل پیگیری

ردیف‌های challenge تاریخی و تازه، prediction هر arm برای هر مثال، checkpoint readoutهای
تازه، گزارش integrity، نتیجهٔ gateها و handoff مرحلهٔ بعد داخل bundle اجرا ثبت می‌شوند.
فایل‌های حجیم پس از import در artifact store محلی می‌مانند و گزارش‌های متنی و JSON لازم
کنار همین نوت‌بوک نگهداری می‌شوند.
