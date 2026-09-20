# M2 — اجرای shadow با Q واقعی و دروازهٔ zero-effect

## اجرای انجام‌شده

روی ۲۴ نمونه از cohort آموزش M0، snapshot بومیِ منجمد ساخته شد و سپس:

1. `ProbeNative` بعد از snapshot کامل اجرا شد؛
2. کاندیدها فقط از `edge_evidence` و top-k بومی انتخاب شدند؛ هیچ gold edge وارد
   انتخاب patch نشد؛
3. Q از `event_source_logits`، `event_target_logits` و
   `event_relation_logits` ساخته شد؛
4. جرم صفر و self/padding با `prepare_patch` حذف شد؛
5. حل‌گر مستقل phase برای هر world و با `nu=0` اجرا شد.

## نتیجه

- همهٔ نمونه‌ها Q detached داشتند.
- خطای نرمال‌سازی `pi` روی یال‌های دارای جرم پشتیبانی حداکثر
  `1.2e-7` بود.
- native geometry فعال نشد و answer head از شاخهٔ shadow فراخوانی نشد.
- خروجی کامل native قبل و بعد از مسیر shadow دقیقاً برابر بود.
- وزن‌های native قبل و بعد دقیقاً برابر ماندند.

این نتیجه یک **آزمون جداسازی و زمان‌بندی** است، نه ادعای بهبود پاسخ یا پذیرش
کالیبراسیون. بنابراین `M2_complete=false` و `M3_authorized_by_results=false`
عمداً حفظ شده‌اند.

جزئیات ماشینی در [m2_q_shadow.json](../manifests/m2_q_shadow.json) و اجرای قابل
تکرار در `scripts/run_m2_q_shadow.py` ثبت شده است.
