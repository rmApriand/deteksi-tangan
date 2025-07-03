import cv2
import mediapipe as mp

# Inisialisasi modul MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

# Inisialisasi deteksi tangan dengan MediaPipe
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Membalik frame agar tidak terbalik (mirror)
        frame = cv2.flip(frame, 1)

        # Mengubah warna frame ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi tangan
        results = hands.process(rgb_frame)

        # Jika ada tangan yang terdeteksi, gambar titik-titik dan sambungan pada tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Tampilkan frame
        cv2.imshow('Hand Detection', frame)

        # Tekan 'q' untuk keluar dari jendela
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()