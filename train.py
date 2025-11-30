import gymnasium as gym
from ale_py import ALEInterface
import ale_py
gym.register_envs(ale_py)
import torch as T
import numpy as np
import argparse
import random
import os
from gymnasium.wrappers import RecordVideo

from utils import (
    FramePreprocessor,
    FrameStack,
    guardar_checkpoint,
    graficar_curvas_entrenamiento,
    mostrar_video,
)
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent, PPOMemory

def get_device(device_arg):
    if device_arg == 'cuda' and T.cuda.is_available():
        return T.device("cuda:0")
    elif device_arg == 'cuda':
        print("Advertencia: Se solicitó CUDA (--device cuda) pero no está disponible. Usando CPU en su lugar.")
        return T.device("cpu")
    elif device_arg == 'cpu':
        return T.device("cpu")
    else:  # auto
        return T.device("cuda:0" if T.cuda.is_available() else "cpu")

def train_dqn(
    dispositivo,
    num_episodios: int = 1000,
    max_pasos_por_episodio: int = 10000,
    pasos_exploracion_inicial: int = 10000,
    intervalo_registro: int = 10,
    intervalo_checkpoint: int = 100,
    directorio_checkpoints: str = 'checkpoints',
    semilla: int = 42
):
    random.seed(semilla)
    np.random.seed(semilla)
    T.manual_seed(semilla)
    if T.cuda.is_available():
        T.cuda.manual_seed(semilla)

    env = gym.make("ALE/Frogger-v5", render_mode="rgb_array")
    env.action_space.seed(semilla)

    preprocesador = FramePreprocessor()
    pila_frames = FrameStack(num_frames=4)

    agente = DQNAgent(
        num_acciones=env.action_space.n,
        dispositivo=dispositivo
    )

    recompensas_episodio = []
    perdidas_episodio = []
    longitudes_episodio = []
    epsilons = []

    print("Iniciando Entrenamiento DQN en Frogger")
    print(f"Dispositivo: {dispositivo}")
    print(f"Número de episodios: {num_episodios}")

    pasos_totales = 0

    print(f"\nFase 1: Exploración inicial ({pasos_exploracion_inicial} pasos)...")
    obs, info = env.reset(seed=semilla)
    frame_preprocesado = preprocesador.preprocesar(obs)
    pila_frames.reiniciar(frame_preprocesado)
    estado = pila_frames.obtener_estado()

    pasos_exploracion = 0
    while pasos_exploracion < pasos_exploracion_inicial:
        accion = env.action_space.sample()
        siguiente_obs, recompensa, terminado, truncado, info = env.step(accion)
        done = terminado or truncado
        frame_preprocesado = preprocesador.preprocesar(siguiente_obs)
        pila_frames.agregar(frame_preprocesado)
        siguiente_estado = pila_frames.obtener_estado()
        agente.almacenar_transicion(estado, accion, recompensa, siguiente_estado, done)
        estado = siguiente_estado
        pasos_exploracion += 1
        if done:
            obs, info = env.reset()
            frame_preprocesado = preprocesador.preprocesar(obs)
            pila_frames.reiniciar(frame_preprocesado)
            estado = pila_frames.obtener_estado()

    print(f"Exploración completa. Tamaño del buffer: {len(agente.memoria)}")

    print(f"\nFase 2: Entrenamiento ({num_episodios} episodios)...")

    for episodio in range(num_episodios):
        obs, info = env.reset()
        frame_preprocesado = preprocesador.preprocesar(obs)
        pila_frames.reiniciar(frame_preprocesado)
        estado = pila_frames.obtener_estado()

        recompensa_episodio_actual = 0
        perdida_episodio_actual = 0
        pasos = 0

        for paso in range(max_pasos_por_episodio):
            accion = agente.seleccionar_accion(estado, entrenando=True)
            siguiente_obs, recompensa, terminado, truncado, info = env.step(accion)
            done = terminado or truncado
            frame_preprocesado = preprocesador.preprocesar(siguiente_obs)
            pila_frames.agregar(frame_preprocesado)
            siguiente_estado = pila_frames.obtener_estado()
            agente.almacenar_transicion(estado, accion, recompensa, siguiente_estado, done)
            perdida = agente.paso_entrenamiento()
            recompensa_episodio_actual += recompensa
            perdida_episodio_actual += perdida
            pasos += 1
            pasos_totales += 1
            estado = siguiente_estado
            if done:
                break

        agente.actualizar_epsilon()
        agente.episodios_realizados += 1

        recompensas_episodio.append(recompensa_episodio_actual)
        perdidas_episodio.append(perdida_episodio_actual / max(pasos, 1))
        longitudes_episodio.append(pasos)
        epsilons.append(agente.epsilon)

        if (episodio + 1) % intervalo_registro == 0:
            recompensa_promedio = np.mean(recompensas_episodio[-intervalo_registro:])
            print(f"Episodio {episodio + 1}/{num_episodios} | Recompensa Promedio: {recompensa_promedio:.2f}")

        if (episodio + 1) % intervalo_checkpoint == 0:
            guardar_checkpoint(agente.red_politica, f"dqn_frogger_episodio_{episodio + 1}.pt", directorio_checkpoints)
            agente.guardar(os.path.join(directorio_checkpoints, f"dqn_agente_episodio_{episodio + 1}.pt"))

    print("\nEntrenamiento Completo!")
    guardar_checkpoint(agente.red_politica, "dqn_frogger_final.pt", directorio_checkpoints)
    agente.guardar(os.path.join(directorio_checkpoints, "dqn_agente_final.pt"))

    graficar_curvas_entrenamiento(recompensas_episodio, perdidas_episodio, epsilons)
    env.close()

def train_ppo(
    dispositivo,
    num_episodios: int = 1000,
    max_pasos_por_episodio: int = 2048,
    intervalo_registro: int = 10,
    intervalo_checkpoint: int = 100,
    directorio_checkpoints: str = 'checkpoints',
    semilla: int = 42
):
    random.seed(semilla)
    np.random.seed(semilla)
    T.manual_seed(semilla)
    if T.cuda.is_available():
        T.cuda.manual_seed(semilla)

    env = gym.make("ALE/Frogger-v5", render_mode="rgb_array")
    env.action_space.seed(semilla)

    preprocesador = FramePreprocessor()
    pila_frames = FrameStack(num_frames=4)

    agente = PPOAgent(num_acciones=env.action_space.n, dispositivo=dispositivo)
    memoria = PPOMemory(max_pasos_por_episodio, 1, dispositivo)

    recompensas_episodio = []
    perdidas_episodio = []
    longitudes_episodio = []

    print("Iniciando Entrenamiento PPO en Frogger")
    print(f"Dispositivo: {dispositivo}")
    print(f"Número de episodios: {num_episodios}")

    for episodio in range(num_episodios):
        obs, info = env.reset(seed=semilla + episodio)
        frame_preprocesado = preprocesador.preprocesar(obs)
        pila_frames.reiniciar(frame_preprocesado)
        estado = pila_frames.obtener_estado()
        
        recompensa_episodio_actual = 0
        done = False
        pasos = 0
        
        while not done:
            accion, log_prob, valor = agente.seleccionar_accion(estado)
            siguiente_obs, recompensa, terminado, truncado, info = env.step(accion)
            done = terminado or truncado
            
            recompensa_episodio_actual += recompensa
            
            frame_preprocesado = preprocesador.preprocesar(siguiente_obs)
            pila_frames.agregar(frame_preprocesado)
            siguiente_estado = pila_frames.obtener_estado()
            
            memoria.agregar(estado, accion, log_prob, valor, recompensa, done)
            
            estado = siguiente_estado
            pasos += 1

            if len(memoria.estados) >= max_pasos_por_episodio:
                perdida = agente.paso_entrenamiento(memoria)
                memoria.reset()
                perdidas_episodio.append(perdida)

        recompensas_episodio.append(recompensa_episodio_actual)
        longitudes_episodio.append(pasos)

        if (episodio + 1) % intervalo_registro == 0:
            recompensa_promedio = np.mean(recompensas_episodio[-intervalo_registro:])
            print(f"Episodio {episodio + 1}/{num_episodios} | Recompensa Promedio: {recompensa_promedio:.2f}")

        if (episodio + 1) % intervalo_checkpoint == 0:
            guardar_checkpoint(agente.actor_critic, f"ppo_frogger_episodio_{episodio + 1}.pt", directorio_checkpoints)

    print("\nEntrenamiento Completo!")
    guardar_checkpoint(agente.actor_critic, "ppo_frogger_final.pt", directorio_checkpoints)
    
    # PPO no tiene una curva de épsilon, así que pasamos una lista vacía
    graficar_curvas_entrenamiento(recompensas_episodio, perdidas_episodio, [])
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento de agentes de RL para Frogger.')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ppo'], help="Agente a entrenar: 'dqn' o 'ppo'.")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="Dispositivo a usar.")
    parser.add_argument('--num_episodes', type=int, default=1000, help="Número de episodios para entrenar.")
    args = parser.parse_args()

    dispositivo = get_device(args.device)

    if args.agent == 'dqn':
        train_dqn(dispositivo=dispositivo, num_episodios=args.num_episodes)
    elif args.agent == 'ppo':
        train_ppo(dispositivo=dispositivo, num_episodios=args.num_episodes)
