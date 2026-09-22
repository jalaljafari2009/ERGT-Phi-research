# Audit relation-target shortcuts and measure frozen native answer/path targets

Experiment: `M2-E004/v001` · stage `M2` · kind `research`.

Hypothesis: If the auxiliary relation target is too direct, a frozen Q-only readout will lose substantial balanced accuracy when the relation channels are zeroed or cyclically permuted, while raw-only access and literal input-token exposure will explain why perfect relation decoding does not establish answer/path improvement. A frozen native answer/path panel is the more relevant target for the next M2 design.

Goal reference: docs/MATHEMATICAL_SPEC.md sections 5, 14, 15 and 17; docs/SPEC_Q_ANCHOR_ADDENDUM.md; docs/OPERATIONAL_ROADMAP_FA.md; ADR-0011

- [Preregistered protocol](protocol.json)
- Notebook: `experiment.ipynb` (generate with the workflow CLI).
- Immutable release lock: `package.json` (created when packaged).
- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.
- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.

An execution success is not a scientific pass. Review gates and limitations before proceeding.

## طراحی قفل‌شده

این ممیزی همان cohort و دو پنجرهٔ pair-disjoint پذیرفته‌شده در M2-E003/v004 را
دوباره می‌سازد. checkpoint مرجع، phase، Q-only و no-phase فقط خوانده می‌شوند و hash
آن‌ها پیش و پس از اجرا کنترل می‌شود. سه مداخلهٔ `full`، `zero` و `permute` فقط روی
شش کانال انتهایی Q در inference اعمال می‌شوند؛ permutation چرخه‌ای `[1,2,0]` در دو
سه‌تاییِ source و target مستقل است.

کنترل raw-only فقط `Psi0` را می‌بیند و دقیقاً سه epoch روی fit تاریخی M0 آموزش
می‌بیند. cohort تازه صرفاً برای ارزیابی است. metricهای relation شامل CE، accuracy،
balanced accuracy، recall هر کلاس، confusion matrix، افت مداخله و نرخ تغییر پیش‌بینی
هستند.

## ممیزی target

runner بررسی می‌کند آیا relation surface دقیقاً در `event_anchor_position` متن خام
وجود دارد و آیا relation sequence در دو عضو counterfactual ثابت می‌ماند، در حالی که
پاسخ درست عوض می‌شود. سپس مدل native منجمد را با metricهای answer accuracy، pair
exact، program-event recall/precision، event-chain exact، selected-event recall و
path coverage می‌سنجد. این پنل به پاسخ و انتخاب مسیر نزدیک‌تر است، اما در این نسخه
هیچ phase adapter تازه‌ای روی آن آموزش نمی‌بیند.

تمام gateهای پروتکل مربوط به کامل‌بودن اجرا، استقلال داده و تغییرنکردن ورودی‌ها هستند.
این آزمایش اجازهٔ M3، بازکردن M8، ادعای بهبود پاسخ یا ادعای برتری Phase نمی‌دهد.
