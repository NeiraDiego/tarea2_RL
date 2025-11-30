import torch as T
import numpy as np
from collections import deque
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def mostrar_video(ruta):
    import subprocess
    archivos_video = list(Path(ruta).glob("*.mp4"))
    if not archivos_video:
        print(f"No se encontraron videos en: {ruta}")
        return
    reproductores = ['xdg-open', 'vlc', 'mpv', 'ffplay']
    for video in archivos_video:
        for reproductor in reproductores:
            try:
                subprocess.Popen([reproductor, str(video)],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                print(f"Abriendo video con {reproductor}: {video}")
                return
            except FileNotFoundError:
                continue
        print(f"No se encontró reproductor. Video guardado en: {video.absolute()}")
        print("Instala un reproductor: sudo apt-get install vlc")

def guardar_checkpoint(modelo, nombre_archivo, directorio_checkpoints='checkpoints'):
    os.makedirs(directorio_checkpoints, exist_ok=True)
    ruta_checkpoint = os.path.join(directorio_checkpoints, nombre_archivo)
    T.save(modelo.state_dict(), ruta_checkpoint)
    print(f"Checkpoint guardado en {ruta_checkpoint}")
    return ruta_checkpoint

def graficar_curvas_entrenamiento(recompensas, perdidas, epsilons, ruta_guardado='curvas_entrenamiento.png'):
    fig, ejes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Métricas de Entrenamiento', fontsize=16)

    ejes[0].plot(recompensas, alpha=0.6, label='Recompensa de Episodio')
    if len(recompensas) >= 10:
        ventana = 10
        promedio_movil = np.convolve(recompensas, np.ones(ventana)/ventana, mode='valid')
        ejes[0].plot(range(ventana-1, len(recompensas)), promedio_movil,
                    'r-', linewidth=2, label=f'Promedio Móvil ({ventana} Episodios)')
    ejes[0].set_ylabel('Recompensa Total')
    ejes[0].set_title('Recompensas de Episodios')
    ejes[0].legend()
    ejes[0].grid(True, alpha=0.3)

    ejes[1].plot(perdidas, alpha=0.6, label='Pérdida Promedio')
    if len(perdidas) >= 10:
        ventana = 10
        promedio_movil_perdida = np.convolve(perdidas, np.ones(ventana)/ventana, mode='valid')
        ejes[1].plot(range(ventana-1, len(perdidas)), promedio_movil_perdida,
                    'r-', linewidth=2, label=f'Promedio Móvil ({ventana} Episodios)')
    ejes[1].set_ylabel('Pérdida')
    ejes[1].set_title('Pérdida de Entrenamiento')
    ejes[1].legend()
    ejes[1].grid(True, alpha=0.3)

    ejes[2].plot(epsilons, 'g-', label='Epsilon')
    ejes[2].set_xlabel('Episodio')
    ejes[2].set_ylabel('Epsilon')
    ejes[2].set_title('Tasa de Exploración (Epsilon)')
    ejes[2].legend()
    ejes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
    print(f"Curvas de entrenamiento guardadas en {ruta_guardado}")
    plt.show()

class FramePreprocessor:
    def __init__(self):
        pass
    def preprocesar(self, frame: np.ndarray) -> np.ndarray:
        gris = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        redimensionado = self._redimensionar_frame(gris, (84, 84))
        normalizado = redimensionado.astype(np.float32) / 255.0
        return normalizado

    def _redimensionar_frame(self, frame: np.ndarray, tamanio: tuple) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(frame, tamanio[::-1], interpolation=cv2.INTER_AREA)
        except ImportError:
            h, w = frame.shape
            altura_objetivo, ancho_objetivo = tamanio
            paso_h = h // altura_objetivo
            paso_w = w // ancho_objetivo
            return frame[::paso_h, ::paso_w][:altura_objetivo, :ancho_objetivo]

class FrameStack:
    def __init__(self, num_frames: int = 4, forma_frame: tuple = (84, 84)):
        self.num_frames = num_frames
        self.forma_frame = forma_frame
        self.frames = deque(maxlen=num_frames)

    def reiniciar(self, frame_inicial: np.ndarray):
        for _ in range(self.num_frames):
            self.frames.append(frame_inicial)

    def agregar(self, frame: np.ndarray):
        self.frames.append(frame)

    def obtener_estado(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.float32)
