import face_recognition
import time
from deepface import DeepFace

class SecurityLogic: 
    def __init__(self, owner_path):
        print(f"--- Initializing Identity Database: {owner_path} ---")
        owner_image = face_recognition.load_image_file(owner_path)
        encodings = face_recognition.face_encodings(owner_image)

        if len(encodings) > 0: 
            self.owner_encoding = encodings[0]
        else: 
            raise ValueError(f"Could not find a face in {owner_path}")
        
        self.unknown_start_time = None
        self.loitering_limit = 10

    
    def analyze_face(self, frame, face_location, face_encoding):
        match = face_recognition.compare_faces([self.owner_encoding], face_encoding, tolerance=0.5)
        
        if True in match: 
            self.unknown_start_time = None
            return "OWNER: AUTHORIZED", (0, 255, 0)
        
        if self.unknown_start_time is None:   
            self.unknown_start_time = time.time()
        
        elapsed = time.time() - self.unknown_start_time

        try: 
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            emotion = analysis[0]['dominant_emotion']
        except: 
            emotion = "neutral"

        if emotion in ['angry', 'fear']:
            return f"UNKNOWN: HOSTILE ({emotion})", (0, 0, 255)
        if elapsed > self.loitering_limit: 
            return f"UNKNOWN: SUSPICIOUS ({int(elapsed)}s)", (0, 165, 255)
        
        return f"UNKNOWN: NEUTRAL ({int(elapsed)}s)", (255, 255, 255)