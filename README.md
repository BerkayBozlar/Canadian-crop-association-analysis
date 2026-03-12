# Kanada Tarım Ürünleri Birliktelik Analizi ve Tahminleme

Bu proje, Kanada'da yetiştirilen tarım ürünlerinin 80 yıllık geçmiş verilerini kullanarak, hangi ürünlerin birlikte ekilme eğiliminde olduğunu analiz etmeyi ve gelecekteki ekim trendlerini tahmin etmeyi amaçlamaktadır.

## Projenin Amaçları ve Kapsamı

Bu çalışmada geleneksel birliktelik analizi bir adım öteye taşınarak zaman serisi mantığıyla dönemsel bir karşılaştırma ve gelecek tahminlemesi yapılmıştır. Temel hedefler şunlardır:

* **Kapsamlı Veri Ön İşleme:** Kanada'nın 80 yıllık karmaşık tarım veri setini, birliktelik analizi algoritmalarının (sepet analizi) çalışabileceği uygun formata getirmek.
* **Dönemsel Trend Karşılaştırması:** 80 yıllık veriyi **ilk 70 yıl** ve **son 10 yıl** olarak ikiye bölerek, tarım ürünlerinin birlikte ekilme alışkanlıklarında zaman içinde yaşanan değişimleri ortaya koymak.
* **Gelecek 10 Yılın Tahminlenmesi (Forecasting):** Elde edilen geçmiş örüntüleri ve kuralları kullanarak, **önümüzdeki 10 yılın her bir yılı için ayrı ayrı** hangi tarım ürünlerinin birlikte ekilme potansiyelinin yüksek olduğunu tahmin etmek.
* **Algoritma Karşılaştırması:** Tüm analiz ve tahmin süreçlerini veri madenciliğinin en popüler iki yöntemi olan **Apriori** ve **FP-Growth** algoritmalarıyla ayrı ayrı çalıştırarak sonuçları ve performansları kıyaslamak.

## Kullanılan Teknolojiler ve Yöntemler

* **Dil:** Python
* **Algoritmalar:** Apriori Algoritması, FP-Growth Algoritması
* **Değerlendirme Metrikleri:** Support (Destek), Confidence (Güven), Lift
* **Veri Görselleştirme:** Elde edilen kuralların ve ürün bazlı destek değerlerinin grafiksel sunumu.

## Proje Çıktıları ve Görseller

Proje kapsamında elde edilen temel çıktılar projedeki klasörler altında sunulmuştur:
* Algoritmaların ürettiği kuralların destek (support) ve güven (confidence) seviyelerini gösteren dağılım grafikleri.
* Dönemsel farklılıkları ortaya koyan karşılaştırmalı analiz tabloları.
* Gelecek 10 yıl için yıl bazlı üretilen tahmin kuralları.
