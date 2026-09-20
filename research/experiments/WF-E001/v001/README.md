# Validate the versioned notebook and result-evidence lifecycle

Experiment: `WF-E001/v001` · stage `workflow` · kind `infrastructure`.

Hypothesis: A deterministic CPU fixture can be packaged, executed, imported and reviewed with exact provenance and all declared artifacts.

Goal reference: research/decisions/ADR-0001.md; docs/OPERATIONAL_ROADMAP_FA.md workflow contract

- [Preregistered protocol](protocol.json)
- Notebook: `experiment.ipynb` (generate with the workflow CLI).
- Immutable release lock: `package.json` (created when packaged).
- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.
- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.

An execution success is not a scientific pass. Review gates and limitations before proceeding.

## اجرای ثبت‌شده

| run | محیط | نتیجه و گزارش |
|---|---|---|
| `local-20260920T111545Z-22c8c9` | Windows / CPU؛ خارج از Colab | [خلاصه](runs/local-20260920T111545Z-22c8c9/summary.md)، [تفسیر](runs/local-20260920T111545Z-22c8c9/interpretation.md)، [داوری](runs/local-20260920T111545Z-22c8c9/reviews/001.json) |

این نمونه تنها گردش کار را بررسی می‌کند. اجرای آن در Colab برای بررسی محیط سرویس
اختیاری است؛ به‌تنهایی پیشرفت علمی M2 محسوب نمی‌شود.

## اجرای همین نسخه در Colab

1. [experiment.ipynb](experiment.ipynb) را در Colab باز کنید؛ CPU کافی است.
2. با Run all، در پنجرهٔ upload دو فایل را با هم انتخاب کنید:
   [package.json](package.json) و ZIP محلی
   `research/packages/WF-E001-v001-015c9a4196a4-source.zip` از ریشهٔ ریپو.
3. Drive را متصل کنید. نوت‌بوک پوشهٔ تازه‌ای در
   `/content/drive/MyDrive/ERGT_Phi/research/WF-E001/v001/<run_id>/` می‌سازد.
4. ZIP نتیجه را از سلول آخر دریافت و در `research/inbox/` قرار دهید؛ مسیر را به ایجنت بدهید.

این ورودی داده یا checkpoint خارجی ندارد. source ZIP دارای ۱۴۶ فایل است؛ قفل و
[source_manifest.json](source_manifest.json) امکان تطبیق و بازیابی نسخهٔ دقیق را می‌دهند.
اگر ZIP محلی موجود نیست، دستور `package` با فایل‌های منبعِ مطابق hash آن را بازسازی می‌کند.
