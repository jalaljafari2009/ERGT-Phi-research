# LEGACY-M2-Q-SHADOW — ممیزی محاسبهٔ جداگانهٔ Q

روی ۲۴ نمونهٔ cohort آموزش، Q از proposal بومی ساخته و phase solver به‌صورت جدا اجرا شد. این اجرا محلی CPU بود؛ نوت‌بوک یا checkpoint تازه برای آن ثبت نشده است.

## نتیجهٔ مشاهده‌شده

Q در همهٔ نمونه‌ها detached بود؛ بیشینهٔ خطای نرمال‌سازی mixture حدود `1.2e-7` بود. خروجی و وزن‌های native در بررسی ثبت‌شده برابر ماندند. geometry فعال نشد و answer head از شاخهٔ shadow فراخوانی نشد. `M2_complete=false` و `M3_authorized_by_results=false` باقی ماندند.

## مدارک

- [گزارش تاریخی](reports/M2_Q_SHADOW_FA.md)، [manifest](manifests/m2_q_shadow.json)، [نتیجهٔ خام](evidence/m2_q_shadow/result.json).
- [آرشیو کامل محلی](../../artifacts/legacy/LEGACY-M2-Q-SHADOW/m2_q_shadow/) یک فایل و ۴۱٬۰۷۳ بایت دارد.
- runner: [run_m2_q_shadow.py](../../../scripts/run_m2_q_shadow.py)، commit `6d669b3fb4d7312832ad2884d3af92647867597f`.

## اصلاح دامنهٔ تفسیر

عنوان تاریخی «دروازهٔ zero-effect» و تعبیر «آزمون زمان‌بندی» نباید به قبولی اتصال واقعی تعمیم داده شوند. ممیزی کد نشان داد این مسیر جداست؛ از `edge_evidence` و top-32 سراسری استفاده می‌کند و معادل `phi` و preview/top-k واقعیِ هر سطر/جهان نیست. برابری خروجی در این مسیر، حفظ gradient/optimizer/RNG در مسیر یکپارچه، اتصال به geometry یا تأخیر واقعی حلقه را ثابت نمی‌کند.

مرحلهٔ بعد [کالیبراسیون Q-conditioned](../LEGACY-M2-Q-CALIBRATION/README.md) بود. الزامات اتصال واقعی در [برنامهٔ عملیاتی](../../../docs/OPERATIONAL_ROADMAP_FA.md) و [بازبینی M2](../../phase_reviews/M2_CURRENT.md) ثبت‌اند.

## ???? ??

[???? ???????? ??](source_provenance.json) hash??? ??????? ?? ?? blob ? commit ???? ??????? Git ??? ??????. ?????? commit ?? ??? ????? ??? ??????? ?????????? ?? ????? ???? ??????? commit ????? ??????. ???? ??? ???? hash ??????? ???? ??? ???? ???? ??? ??? ???.
