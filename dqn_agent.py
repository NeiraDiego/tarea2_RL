import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from utils import FramePreprocessor, FrameStack

class ReplayBuffer:
    def __init__(self, capacidad: int = 50000):
        self.buffer = deque(maxlen=capacidad)
        self.capacidad = capacidad

    def agregar(self, estado: np.ndarray, accion: int, recompensa: float,
             siguiente_estado: np.ndarray, done: bool):
        self.buffer.append((estado, accion, recompensa, siguiente_estado, done))

    def muestrear(self, tamanio_batch: int) -> tuple:
        batch = random.sample(self.buffer, tamanio_batch)

        estados, acciones, recompensas, siguientes_estados, dones = zip(*batch)

        return (
            np.array(estados, dtype=np.float32),
            np.array(acciones, dtype=np.int64),
            np.array(recompensas, dtype=np.float32),
            np.array(siguientes_estados, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_acciones: int = 5):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_acciones)

        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for modulo in self.modules():
            if isinstance(modulo, nn.Conv2d) or isinstance(modulo, nn.Linear):
                nn.init.kaiming_normal_(modulo.weight, nonlinearity='relu')
                if modulo.bias is not None:
                    nn.init.constant_(modulo.bias, 0)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        valores_q = self.fc2(x)

        return valores_q

class DQNAgent:
    def __init__(
        self,
        num_acciones: int = 5,
        tasa_aprendizaje: float = 1e-4,
        gamma: float = 0.99,
        epsilon_inicio: float = 1.0,
        epsilon_fin: float = 0.1,
        decaimiento_epsilon: float = 0.995,
        capacidad_buffer: int = 50000,
        tamanio_batch: int = 32,
        frecuencia_actualizacion_objetivo: int = 1000,
        dispositivo: T.device = T.device("cpu")
    ):
        self.num_acciones = num_acciones
        self.gamma = gamma
        self.epsilon = epsilon_inicio
        self.epsilon_inicio = epsilon_inicio
        self.epsilon_fin = epsilon_fin
        self.decaimiento_epsilon = decaimiento_epsilon
        self.tamanio_batch = tamanio_batch
        self.frecuencia_actualizacion_objetivo = frecuencia_actualizacion_objetivo
        self.dispositivo = dispositivo

        self.red_politica = DQN(num_acciones).to(dispositivo)
        self.red_objetivo = DQN(num_acciones).to(dispositivo)
        self.red_objetivo.load_state_dict(self.red_politica.state_dict())
        self.red_objetivo.eval()

        self.optimizador = optim.Adam(self.red_politica.parameters(), lr=tasa_aprendizaje)
        self.memoria = ReplayBuffer(capacidad=capacidad_buffer)

        self.pasos_realizados = 0
        self.episodios_realizados = 0

    def seleccionar_accion(self, estado: np.ndarray, entrenando: bool = True) -> int:
        if not entrenando or random.random() > self.epsilon:
            with T.no_grad():
                tensor_estado = T.FloatTensor(estado).unsqueeze(0).to(self.dispositivo)
                valores_q = self.red_politica(tensor_estado)
                accion = valores_q.max(1)[1].item()
                return accion
        else:
            return random.randrange(self.num_acciones)

    def almacenar_transicion(self, estado: np.ndarray, accion: int, recompensa: float,
                        siguiente_estado: np.ndarray, done: bool):
        self.memoria.agregar(estado, accion, recompensa, siguiente_estado, done)

    def paso_entrenamiento(self) -> float:
        if len(self.memoria) < self.tamanio_batch:
            return 0.0

        estados, acciones, recompensas, siguientes_estados, dones = self.memoria.muestrear(self.tamanio_batch)

        estados = T.FloatTensor(estados).to(self.dispositivo)
        acciones = T.LongTensor(acciones).to(self.dispositivo)
        recompensas = T.FloatTensor(recompensas).to(self.dispositivo)
        siguientes_estados = T.FloatTensor(siguientes_estados).to(self.dispositivo)
        dones = T.FloatTensor(dones).to(self.dispositivo)

        valores_q_actuales = self.red_politica(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)

        with T.no_grad():
            siguientes_valores_q = self.red_objetivo(siguientes_estados).max(1)[0]
            valores_q_objetivo = recompensas + (1 - dones) * self.gamma * siguientes_valores_q

        perdida = F.smooth_l1_loss(valores_q_actuales, valores_q_objetivo)

        self.optimizador.zero_grad()
        perdida.backward()

        for param in self.red_politica.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        self.optimizador.step()

        self.pasos_realizados += 1

        if self.pasos_realizados % self.frecuencia_actualizacion_objetivo == 0:
            self.red_objetivo.load_state_dict(self.red_politica.state_dict())

        return perdida.item()

    def actualizar_epsilon(self):
        self.epsilon = max(self.epsilon_fin, self.epsilon * self.decaimiento_epsilon)

    def guardar(self, ruta_archivo: str):
        T.save({
            'estado_dict_red_politica': self.red_politica.state_dict(),
            'estado_dict_red_objetivo': self.red_objetivo.state_dict(),
            'estado_dict_optimizador': self.optimizador.state_dict(),
            'epsilon': self.epsilon,
            'pasos_realizados': self.pasos_realizados,
            'episodios_realizados': self.episodios_realizados,
        }, ruta_archivo)

    def cargar(self, ruta_archivo: str):
        checkpoint = T.load(ruta_archivo, map_location=self.dispositivo)
        self.red_politica.load_state_dict(checkpoint['estado_dict_red_politica'])
        self.red_objetivo.load_state_dict(checkpoint['estado_dict_red_objetivo'])
        self.optimizador.load_state_dict(checkpoint['estado_dict_optimizador'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_fin)
        self.pasos_realizados = checkpoint.get('pasos_realizados', 0)
        self.episodios_realizados = checkpoint.get('episodios_realizados', 0)
