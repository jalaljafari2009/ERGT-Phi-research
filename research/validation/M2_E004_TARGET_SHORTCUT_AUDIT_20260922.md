# اعتبارسنجی M2-E004/v001

- spec lock: pass؛ `23450` byte و SHA-256 برابر
  `04ac57cc76395f7b032356bcac38b70b796233ac4aa59ba26a6b2474977c5d90`.
- تست مستقیم تغییر: `pytest tests/test_m2_shortcut_audit.py tests/test_m2_fresh.py -q`؛
  ۹ تست پاس.
- compile: `ergt_phi/m2_shortcut_audit.py` و
  `scripts/run_m2_target_shortcut_audit.py` پاس.
- اجرای package-bound: `execution_completed`، return code صفر، ۲۰/۲۰ gate پاس،
  required artifact مفقود صفر.
- source commit: `e34f7448d9d3fa3f44a822640dd0bca0ca81d4df`.
- package SHA-256: `e452e916fb19e5c5720d718760fb7af54bd43db97d6880111d883a29523d4c5f`.
- bundle SHA-256: `f961ea44cdd74224606c1f33000158a3d82ccd83fc39c2f4035b88858b86fd6e`.
- layout audit: pass؛ ۱۳۴ اصل مهاجرتی، ۱۲۱ شاهد tracked، ۱۲ منبع، ۱۹۵ رکورد
  provenance تاریخی، ۶۲ عضو ZIP، ۶ run sealed و ۶۱۶ پیوند Markdown بررسی شدند؛ خطا صفر.

اجرای بدون محدودکردن discovery به‌اشتباه کپی‌های تاریخی زیر artifacts/workspace را
جمع کرد و معتبر نبود. اجرای `pytest tests`، ۱۲۵ تست را پاس کرد و ۵۵ fixture به‌علت
ACL مسیر موقت میزبان پیش از test body خطا دادند. این خطاها failure علمی/کد محسوب
نشدند و پنهان نشده‌اند؛ تست‌های مستقیم و auditهای نهایی مستقل اجرا و ثبت می‌شوند.

یک invocation اولیهٔ local runner از ریشهٔ Git، پیش از source verification به‌علت نبود
`PACKAGE_MANIFEST.json` متوقف شد؛ این فایل طبق قرارداد داخل source ZIP است. هیچ مدل یا
metric علمی در آن invocation اجرا نشد. اجرای ثبت‌شده از ریشهٔ استخراج‌شدهٔ package با
run-id تازه انجام شد و تنها همان bundle کامل import و review شد.
