# راهنمای ابزارهای پروژه

اسکریپت‌های اجرا در یک پوشهٔ ثابت مانده‌اند تا فرمان‌های قبلی قابل شناسایی باشند.
مستندات، نتایج و checkpointهای آن‌ها در پروندهٔ پژوهشی متناظر دسته‌بندی شده‌اند.

| دسته | ابزار |
|---|---|
| آزمایش جدید و Colab نسخه‌دار | `research.py`؛ راهنما در `research/WORKFLOW.md` |
| فهرست تاریخچه و مقاله | `index_research_history.py`، `export_research_catalog.py` |
| کنترل سامان‌دهی | `audit_research_layout.py` |
| مهاجرت انجام‌شده و قابل ممیزی | `organize_research_history.py`؛ بدون نیاز به اجرای عادی مجدد |
| پذیرش مهندسی M0/M1 | `run_m0.py`، `run_m1.py` |
| آموزش و ممیزی مرجع تک‌بذر | `run_reference_training.py`، `audit_trained_m0.py` |
| runnerهای تاریخی M2 | `run_m2.py`، `run_m2_diagnostic.py`، `run_m2_information_probe.py`، `run_m2_proposal_probe.py`، `run_m2_q_shadow.py`، `run_m2_q_calibration.py` |
| جمع‌بندی ابزارهای قدیمی | خانوادهٔ `finalize_*.py` |
| ساخت بستهٔ سازگاری M0 | `build_colab_notebook.py`، `package_m0.py` |

خواندن ورودی‌های پذیرفته‌شده از طریق `ergt_phi/research_paths.py` و نقشهٔ
`research/legacy/path_map.json` انجام می‌شود. نتایج تازهٔ ابزارهای قدیمی در
`runs/` و `research/workspace/manifests/` تولید می‌شوند و آرشیو را بازنویسی نمی‌کنند.
finalizerها hash سورس را همچنان بررسی می‌کنند؛ نتیجهٔ تاریخی با سورس تازه نباید
دوباره نهایی شود. برای ادامهٔ علمی، آزمایش نسخه‌دار جدید ساخته شود.

نوت‌بوک اصلی M0 در پروندهٔ legacy شاهد تاریخی است. لانچر سازگاری جدید در
`research/workspace/notebook/M0_Reference_Training.ipynb` و ZIP آزمایشی آن در
`research/packages/legacy-M0/ERGT-Phi-M0.zip` ساخته می‌شود. این‌ها جای بستهٔ اصلی
ارسال‌شده توسط پژوهشگر یا بستهٔ قفل‌شدهٔ یک آزمایش جدید نیستند.

`bootstrap_m0.py` ابزار استخراج اولیه است و بازنویسی `native_steps.py` دارد؛
در گردش عادیِ پژوهش به‌عنوان فرمان ادامه استفاده نشود.
