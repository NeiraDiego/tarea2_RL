import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_acciones):
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        
        self.actor = nn.Linear(512, num_acciones)
        self.critic = nn.Linear(512, 1)

        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for modulo in self.modules():
            if isinstance(modulo, nn.Conv2d) or isinstance(modulo, nn.Linear):
                nn.init.orthogonal_(modulo.weight, np.sqrt(2))
                if modulo.bias is not None:
                    nn.init.constant_(modulo.bias, 0)

    def forward(self, x):
        x = T.relu(self.conv1(x))
        x = T.relu(self.conv2(x))
        x = T.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = T.relu(self.fc1(x))
        
        policy_logits = self.actor(x)
        value = self.critic(x)
        
        return policy_logits, value

class PPOAgent:
    def __init__(self, num_acciones, tasa_aprendizaje=2.5e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.1,
                 ppo_epochs=4, minibatch_size=256, entropy_coef=0.01, value_loss_coef=0.5, dispositivo='cpu'):
        self.dispositivo = dispositivo
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.actor_critic = ActorCritic(num_acciones).to(self.dispositivo)
        self.optimizador = optim.Adam(self.actor_critic.parameters(), lr=tasa_aprendizaje, eps=1e-5)

    def seleccionar_accion(self, estado):
        with T.no_grad():
            estado = T.FloatTensor(estado).unsqueeze(0).to(self.dispositivo)
            policy_logits, value = self.actor_critic(estado)
            dist = Categorical(logits=policy_logits)
            accion = dist.sample()
            log_prob = dist.log_prob(accion)
        return accion.item(), log_prob.item(), value.item()

    def paso_entrenamiento(self, memoria):
        # Obtener datos de la memoria
        estados, acciones, log_probs_viejos, valores, recompensas, dones, batches = memoria.generar_batches(self.minibatch_size)

        # Calcular ventajas
        ventajas = np.zeros(len(recompensas), dtype=np.float32)
        ultima_ventaja = 0
        for t in reversed(range(len(recompensas) - 1)):
            if dones[t]:
                ultima_ventaja = 0
            delta = recompensas[t] + self.gamma * valores[t+1] * (1-dones[t+1]) - valores[t]
            ultima_ventaja = delta + self.gamma * self.gae_lambda * (1-dones[t+1]) * ultima_ventaja
            ventajas[t] = ultima_ventaja
        
        ventajas = T.tensor(ventajas).to(self.dispositivo)
        valores = T.tensor(valores).to(self.dispositivo)
        
        # Normalizar ventajas
        ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            for batch in batches:
                estados_batch = T.tensor(estados[batch], dtype=T.float32).to(self.dispositivo)
                acciones_batch = T.tensor(acciones[batch], dtype=T.int64).to(self.dispositivo)
                log_probs_viejos_batch = T.tensor(log_probs_viejos[batch], dtype=T.float32).to(self.dispositivo)
                
                policy_logits, critic_value = self.actor_critic(estados_batch)
                dist = Categorical(logits=policy_logits)
                
                # Pérdida del Actor (Política)
                log_probs_nuevos = dist.log_prob(acciones_batch)
                ratio = (log_probs_nuevos - log_probs_viejos_batch).exp()
                
                surr1 = ratio * ventajas[batch]
                surr2 = T.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * ventajas[batch]
                actor_loss = -T.min(surr1, surr2).mean()
                
                # Pérdida del Crítico (Valor)
                retornos = ventajas[batch] + valores[batch]
                critic_loss = (retornos - critic_value.squeeze()).pow(2).mean()
                
                # Pérdida de Entropía
                entropy_loss = -dist.entropy().mean()
                
                # Pérdida Total
                perdida_total = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
                
                self.optimizador.zero_grad()
                perdida_total.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizador.step()

        return perdida_total.item()

class PPOMemory:
    def __init__(self, num_pasos, num_workers, dispositivo):
        self.num_pasos = num_pasos
        self.num_workers = num_workers
        self.dispositivo = dispositivo
        self.reset()

    def reset(self):
        self.estados = []
        self.acciones = []
        self.log_probs = []
        self.valores = []
        self.recompensas = []
        self.dones = []

    def agregar(self, estado, accion, log_prob, valor, recompensa, done):
        self.estados.append(estado)
        self.acciones.append(accion)
        self.log_probs.append(log_prob)
        self.valores.append(valor)
        self.recompensas.append(recompensa)
        self.dones.append(done)

    def generar_batches(self, minibatch_size):
        n_muestras = len(self.estados)
        indices = np.arange(n_muestras)
        np.random.shuffle(indices)
        
        estados = np.array(self.estados)
        acciones = np.array(self.acciones)
        log_probs = np.array(self.log_probs)
        valores = np.array(self.valores)
        recompensas = np.array(self.recompensas)
        dones = np.array(self.dones)

        batches = [
            indices[i:i + minibatch_size]
            for i in range(0, n_muestras, minibatch_size)
        ]
        
        return estados, acciones, log_probs, valores, recompensas, dones, batches
