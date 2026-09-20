# LEGACY-M2-PROPOSAL — اطلاعات رابطه در ProbeNative

این probe اطلاعات رابطه را از خروجی label-free خودِ native، قبل از حل پاسخ، اندازه گرفت. ویژگی‌ها از logits/mixture رابطه و شواهد زوج/یال ساخته شدند؛ endpoint طلایی پس از forward فقط برای انتخاب رویدادهای loss استفاده شد.

دقت متوازن monitor واقعی `1.0` و کنترل برچسب درهم‌ریخته `0.3333333333333333` بود. CE واقعی monitor برابر `0.034150440245866776` است. این اعداد **دقت پاسخ نهایی ERGT نیستند** و پذیرش M2 یا M3 محسوب نمی‌شوند. نتیجه نشان می‌دهد در همین cohort و طراحی probe، اطلاعات رابطه در proposal قابل استخراج است. کنترل درهم‌ریخته همهٔ انواع نشت/shortcut را به‌تنهایی رد نمی‌کند.

## مدارک و ادامه

- [گزارش](reports/M2_PROPOSAL_PROBE_FA.md)، [manifest نهایی](manifests/m2_proposal.json)، [پروتکل و نتیجهٔ خام](evidence/m2_proposal/).
- [نسخهٔ کامل محلی](../../artifacts/legacy/LEGACY-M2-PROPOSAL/m2_proposal/) شامل دو فایل و ۴٬۹۵۰ بایت است؛ checkpoint و نوت‌بوک تاریخی ندارد.
- runner: [run_m2_proposal_probe.py](../../../scripts/run_m2_proposal_probe.py)، commit `40b6ad4c90850b5e268983cb8b0e14dbafe4dbe3`.

این یافته جهت طراحی را به استفاده از Q بومی برد. دو پیگیری مرتبط عبارت‌اند از [قرارداد تأخیر Q](../LEGACY-M2-Q-TIMING/README.md) و [محاسبهٔ جداگانهٔ Q-shadow](../LEGACY-M2-Q-SHADOW/README.md). عبارت‌های پیشنهادی گزارش دربارهٔ زمان‌بندی، برنامهٔ طراحی‌اند؛ اجرای حلقهٔ یکپارچه از این probe اثبات نمی‌شود.

## ???? ??

[???? ???????? ??](source_provenance.json) hash??? ??????? ?? ?? blob ? commit ???? ??????? Git ??? ??????. ?????? commit ?? ??? ????? ??? ??????? ?????????? ?? ????? ???? ??????? commit ????? ??????. ???? ??? ???? hash ??????? ???? ??? ???? ???? ??? ??? ???.
