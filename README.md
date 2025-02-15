# پروژه طبقه بندی تصاویر

### نمای کلی
در این پروژه یک مدل طبقه بندی تصاویر طراحی کردم که با استفاده از اسکرپینگ تصاویر موجود در سایت دیوار و ایجاد یک دیتاست در 11 کلاس اقدام به طبقه بندی آنها می کند.
هدف اصلی از انجام این پروژه توسعه مدلی است که بتواند تصاویر را به‌طور خودکار شناسایی و در یکی از 11 کلاس مشخص شده طبقه‌بندی کند. این نوع مدل‌ها در بسیاری از صنایع کاربرد دارند، از جمله در تحلیل تصاویر پزشکی، تشخیص اشیاء ، سیستم های نظارتی، اپلیکیشن های شبیه سازی واقعیت افزوده و شناسایی تصاویر مختلف در رسانه‌ها.


### دیتاست
دیتاست استفاده شده شامل حدود 12هزار تصویر که با استفاده از وب اسکرپینگ از سایت دیوار(https://divar.ir) استخراج شده و در 11 کلاس به نام های  206, bird, carpet, cat, cellphone, dog, fish, horse, laptop, perfume, sheap قرار گرفته است.
### پیش پردازش داده ها
#### بارگذاری داده‌ها از پوشه‌ها:
تصاویر تمامی پوشه‌های موجود در دیتاست بارگذاری می‌شوند.
سپس، تصاویر از فرمت‌های مختلف به فرمت NumPy array تبدیل می‌شوند.
#### تغییر اندازه تصاویر:
هر تصویر به اندازه (64, 64) تغییر اندازه داده می‌شود تا اندازه‌ها برای ورودی به مدل ثابت و یکسان باشند.
#### نرمال‌سازی پیکسل‌ها:
تصاویر به مقدار 255 تقسیم می‌شوند تا مقدار پیکسل‌ها در بازه [0, 1] قرار گیرند، که کمک می‌کند تا مدل عملکرد بهتری داشته باشد.
#### برچسب‌گذاری (Label Encoding):
برچسب‌های تصاویر که شامل نام پوشه‌های مربوطه است به صورت عددی باینری با استفاده از    LabelBinarizerتبدیل می‌شوند.
هر برچسب به یک بردار باینری تبدیل می‌شود که طول آن برابر با تعداد کلاس‌ها است و برای هرنمونه تنها یک مقدار 1 وجود دارد.
#### تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی:
داده‌ها با استفاده از train_test_split به نسبت 80% برای آموزش و 20% برای آزمون تقسیم می‌شوند.
این پیش‌پردازش‌ها به مدل کمک می‌کنند تا از داده‌های ورودی به طور موثر استفاده کند


### ساخت مدل 
برای ساخت مدل، از یک شبکه عصبی کانولوشنی (CNN) استفاده شده است. مدل به طور سلسله‌مراتبی شامل لایه‌های مختلفی است که هرکدام به بهبود عملکرد مدل کمک می‌کنند. در اینجا، لایه‌ها به شرح زیر تعریف شده‌اند:
Convolutional Layers
در ابتدا، چهار لایه کانولوشن به مدل افزوده شده‌اند که هرکدام از فیلترهایی به اندازه (3, 3) برای استخراج ویژگی‌های مختلف تصاویر استفاده می‌کنند.
هر لایه کانولوشن با استفاده از تابع فعال‌سازی ReLU عمل می‌کند که به مدل اجازه می‌دهد تا ویژگی‌های پیچیده‌تری را یاد بگیرد.
Pooling Layers
پس از هر لایه کانولوشن، یک لایه  MaxPool به مدل اضافه شده است که اندازه ویژگی‌ها را کاهش می‌دهد و مدل را کارآمدتر می‌کند.
 Batch Normalization
از لایه‌های نرمال‌سازی برای تنظیم داده‌ها و جلوگیری از مشکلات ناشی از نوسانات زیاد داده‌ها استفاده شده است.
Dropout
برای جلوگیری از overfitting، از لایه‌های Dropout برای قطع درصدی از اتصالات در حین آموزش استفاده شده است.
Dense
پس از لایه‌های مربوط به مرحله feature extracion، داده‌ها به یک لایه Dense متصل می‌شوند که از یک تابع فعال‌سازی ReLU استفاده می‌کند.
در نهایت، یک لایه Dense دیگر با تابع فعال‌سازی softmax برای پیش‌بینی نهایی خروجی مدل (که تعداد کلاس‌ها را نشان می‌دهد) اضافه شده است.


### آموزش مدل
مدل ساخته شده پس از آماده‌سازی داده‌ها (که قبلاً در بخش پیش‌پردازش توضیح داده شد) برای آموزش استفاده می‌شود. برای آموزش مدل از داده‌های آموزشی و آزمایشی استفاده می‌شود و فرآیند آموزش شامل تنظیمات زیر است:
#### تعریف پارامترهای آموزش:
داده‌های آموزشی x_train و y_trainو داده‌های آزمایشی x_test و  y_testبرای آموزش و ارزیابی مدل استفاده می‌شوند.
تعداد ایپاک‌ها برابر با 20 و  batch sizeبرابر با 32 تنظیم شده است.
#### کاهش خطا: (Loss) 
از تابع categorical_crossentropy  برای محاسبه خطای مدل در هر پیش‌بینی استفاده می‌شود.
مدل با استفاده از الگوریتم بهینه‌سازی Adam  آموزش داده می‌شود.
#### نظارت بر عملکرد مدل:
در حین آموزش، Accuracyو   Lossبر اساس داده‌های آزمایشی نیز نمایش داده می‌شود تا از overfitting جلوگیری شود.
#### نمایش نتایج:
برای ارزیابی عملکرد مدل، از نمودارهای مختلف برای نمایش صحت و خطای مدل در طول دوره‌های مختلف آموزش استفاده شده است. این نمودارها به ما کمک می‌کنند تا بفهمیم مدل در حال یادگیری است یا به overfitting دچار شده است.


### ذخیره مدل
پس از آموزش مدل، برای استفاده در آینده، مدل آموزش‌دیده در یک فایل با فرمت keras ذخیره می‌شود تا بتوان از آن برای پیش‌بینی‌های بعدی استفاده کرد.


### نتیجه‌گیری
در این پروژه، یک مدل شبکه عصبی کانولوشنی (CNN) برای طبقه‌بندی تصاویر از یک دیتاست با 11 کلاس مختلف پیاده‌سازی و آموزش داده شد. هدف این پروژه پیش‌بینی و دسته‌بندی صحیح تصاویر به کلاس‌های مربوطه با استفاده از روش‌های پیشرفته یادگیری عمیق بود.
نتایج بدست‌آمده نشان می‌دهند که مدل توانسته است به دقت قابل توجهی در شبیه‌سازی رفتار صحیح برای طبقه‌بندی تصاویر برسد. در طول فرآیند آموزش، از تکنیک‌هایی مانند Dropout و Batch Normalization برای جلوگیری از overfitting و بهبود دقت استفاده شد.
در نهایت، مدل توانست ویژگی‌های پیچیده تصاویر را یاد بگیرد و دقت بالایی در شبیه‌سازی پیش‌بینی‌ها در داده‌های آزمایشی نشان داد. از طریق استفاده از روش‌هایی مانند ویژگی‌های استانداردسازی و آموزش مدل با تعداد ایپاک‌های مختلف، عملکرد مدل به میزان قابل توجهی بهبود یافت.
این پروژه نشان می‌دهد که چگونه می‌توان با استفاده از CNN، مشکلات پیچیده طبقه‌بندی تصاویر را حل کرده و از ویژگی‌های استخراج‌شده برای پیش‌بینی‌های دقیق‌تر استفاده کرد. همچنین، فرآیند آموزش و پیش‌پردازش داده‌ها به وضوح نشان‌دهنده چگونگی بهینه‌سازی مدل‌های یادگیری عمیق برای داده‌های مختلف است.

