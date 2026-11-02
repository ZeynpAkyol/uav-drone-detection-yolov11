from ultralytics import YOLO
import cv2
import os
import glob

model_path = 'runs/detect/v11_50epoch/weights/best.pt'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_kontrol():
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' bulunamadı! Lütfen model yolunu kontrol et.")
        return None
    return YOLO(model_path)

def image_test():
    model = model_kontrol()
    if model is None:
        return
    image_dir = os.path.join(BASE_DIR, "image")
    images = glob.glob(os.path.join(image_dir, "*.jpg"))

    if not images:
        print(f"UYARI: '{image_dir}' klasöründe hiç .jpg uzantılı resim bulunamadı!")
        return

    print(f"{len(images)} adet görsel işleniyor...")
    model.predict(
        source=images,
        conf=0.40,
        show=True,  
        save=True
    )
    print("Resim testleri tamamlandı. Sonuçlar 'runs/predict' altında saklandı.")


def video_test():
    model = model_kontrol()
    if model is None:
        return

    video_dir = os.path.join(BASE_DIR, "video")
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if not video_paths:
        print(f"UYARI: '{video_dir}' klasöründe hiç .mp4 video bulunamadı!")
        return

    for video_dosyasi in video_paths:
        print(f"İşleniyor: {video_dosyasi}")
        
        try:
            model.predict(
                source=video_dosyasi,
                conf=0.45,
                show=True,
                save=True
            )
        except Exception as e:
            print(f"Hata oluştu ({video_dosyasi}): {e}")


def camera_test():
    model = model_kontrol()
    if model is None:
        return

    print("Kamera açılıyor... (Çıkmak için klavyeden 'q' tuşuna basın)")

    results = model.predict(
        source=0,        
        conf=0.6,
        show=True,
        stream=True,
        device=0
    )

    for r in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    while True:
        print("""
=====================================
     YOLOv11 Hava Araçları TEST
=====================================
1 - Resim Testi
2 - Canlı Kamera Testi
3 - Video Testi       
4 - Çıkış
""")

        secim = input("Seçiminizi girin (1/2/3/4): ").strip()

        if secim == "1":
            image_test()
        elif secim == "2":
            camera_test()
        elif secim == "3":
            video_test()
        elif secim == "4":
            print("Programdan çıkılıyor. İyi günler!")
            break
        else:
            print("Geçersiz seçim! Lütfen 1, 2, 3 veya 4 girin.")


if __name__ == "__main__":
    main()