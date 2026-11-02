from ultralytics import YOLO
import cv2
import os
import glob
model_path = 'runs/detect/v11_50epoch/weights/best.pt'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_kontrol():
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' bulunamadı!")
        return None
    return YOLO(model_path)

def image_test():
    model = model_kontrol()
    if model is None:
        return

    images = glob.glob(os.path.join(os.path.join(BASE_DIR, "image"), "*.jpg"))

    for img in images:
        if not os.path.exists(img):
            print(f"Bulunamadı: {img}")
            return

    model.predict(
        source=images,
        conf=0.40,
        show=False,
        save=True
    )


def video_test():

    model = model_kontrol()
    if model is None:
        return

    video_paths = glob.glob(os.path.join(os.path.join(BASE_DIR, "video"), "*.mp4"))
    
    gecerli_videolar = []
    
    for video in video_paths:
        if os.path.exists(video):
            gecerli_videolar.append(video)
        else:
            print(f"UYARI: Video bulunamadı ve atlanacak: {video}")

    if not gecerli_videolar:
        print("Hiçbir video dosyası bulunamadı!")
        return

    for video_dosyasi in gecerli_videolar:
        print(f"İşleniyor: {video_dosyasi}")
        
        try:
            model.predict(
                source=video_dosyasi,
                conf=0.45,
                show=False,
                save=True
            )
        except Exception as e:
            print(f"Hata oluştu ({video_dosyasi}): {e}")


def camera_test():
    model = model_kontrol()
    if model is None:
        return

    print("Kamera açılıyor (Çıkmak için 'q')")

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
    print("""
 YOLOv11 Hava Araçları TEST

1 - Resim Testi
2 - Canlı Kamera Testi
3 - Video Testi      
4 - Çıkış
""")

    secim = input("Seçiminizi girin (1/2/3): ")

    if secim == "1":
        image_test()
    elif secim == "2":
        camera_test()
    elif secim == "3":
        video_test()
    elif secim == "4":
        print("Çıkılıyor...")
    else:
        print("Geçersiz seçim!")


if __name__ == "__main__":
    main()
