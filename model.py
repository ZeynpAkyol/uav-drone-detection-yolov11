from roboflow import Roboflow
from ultralytics import YOLO

def main():

    rf = Roboflow(api_key="rWQkrN0Q7HK4LS8VAOPo")
    project = rf.workspace("mediha-oyget").project("hava_araclari_tespit-fjxzq")
    version = project.version(1)
    dataset = version.download("yolov11")

    model = YOLO('yolo11m.pt')  
    
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        
        epochs=50,     
        imgsz=640,      
        batch=32,   
        workers=8,
        amp=True,      
        name='hava_araclari_modeli', 
        device=0,
        cache='disk'
    )
    
if __name__ == '__main__':
    main()