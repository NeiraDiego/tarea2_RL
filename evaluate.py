import gymnasium as gym
from ale_py import ALEInterface
import ale_py
gym.register_envs(ale_py)
import torch as T
import numpy as np
import argparse
from gymnasium.wrappers import RecordVideo

from utils import FramePreprocessor, FrameStack, mostrar_video
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent

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

def evaluar_agente(agente, dispositivo, num_episodios: int = 10, renderizar: bool = False,
                   grabar_video: bool = True, directorio_video: str = './videos_evaluacion'):
    if grabar_video:
        env = gym.make("ALE/Frogger-v5", render_mode="rgb_array")
        env = RecordVideo(env, directorio_video, episode_trigger=lambda e: True)
    else:
        modo_renderizado = "human" if renderizar else "rgb_array"
        env = gym.make("ALE/Frogger-v5", render_mode=modo_renderizado)

    preprocesador = FramePreprocessor()
    pila_frames = FrameStack(num_frames=4)

    recompensas_episodio = []
    longitudes_episodio = []

    print(f"\nEvaluando agente por {num_episodios} episodios...")

    for episodio in range(num_episodios):
        obs, info = env.reset()
        frame_preprocesado = preprocesador.preprocesar(obs)
        pila_frames.reiniciar(frame_preprocesado)
        estado = pila_frames.obtener_estado()

        recompensa_episodio_actual = 0
        pasos = 0
        done = False

        while not done:
            if isinstance(agente, DQNAgent):
                accion = agente.seleccionar_accion(estado, entrenando=False)
            elif isinstance(agente, PPOAgent):
                accion, _, _ = agente.seleccionar_accion(estado)
            else:
                raise TypeError("Tipo de agente no soportado para evaluación.")

            siguiente_obs, recompensa, terminado, truncado, info = env.step(accion)
            done = terminado or truncado

            frame_preprocesado = preprocesador.preprocesar(siguiente_obs)
            pila_frames.agregar(frame_preprocesado)
            estado = pila_frames.obtener_estado()

            recompensa_episodio_actual += recompensa
            pasos += 1

        recompensas_episodio.append(recompensa_episodio_actual)
        longitudes_episodio.append(pasos)
        print(f"Episodio {episodio + 1}: Recompensa = {recompensa_episodio_actual:.1f}, Longitud = {pasos}")

    env.close()

    print("\nResultados de Evaluación:")
    print(f"  Recompensa Media: {np.mean(recompensas_episodio):.2f} ± {np.std(recompensas_episodio):.2f}")

    if grabar_video:
        print(f"\nVideos guardados en {directorio_video}")
        mostrar_video(directorio_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluación de agentes de RL para Frogger.')
    parser.add_argument('--agent', type=str, required=True, choices=['dqn', 'ppo'], help="Agente a evaluar: 'dqn' o 'ppo'.")
    parser.add_argument('--model_path', type=str, required=True, help="Ruta al modelo entrenado (.pt).")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="Dispositivo a usar.")
    parser.add_argument('--episodes', type=int, default=5, help="Número de episodios para evaluar.")
    args = parser.parse_args()

    dispositivo = get_device(args.device)
    
    env = gym.make("ALE/Frogger-v5")
    num_acciones = env.action_space.n
    env.close()

    agente = None
    if args.agent == 'dqn':
        agente = DQNAgent(num_acciones=num_acciones, dispositivo=dispositivo)
        try:
            # Intenta cargar como un checkpoint de agente completo
            agente.cargar(args.model_path)
            print(f"Checkpoint de agente DQN cargado desde {args.model_path}")
        except KeyError:
            # Si falla, carga solo el state_dict de la red
            agente.red_politica.load_state_dict(T.load(args.model_path, map_location=dispositivo))
            print(f"State dict de red DQN cargado desde {args.model_path}")
    elif args.agent == 'ppo':
        agente = PPOAgent(num_acciones=num_acciones, dispositivo=dispositivo)
        agente.actor_critic.load_state_dict(T.load(args.model_path, map_location=dispositivo))
        print(f"State dict de red PPO cargado desde {args.model_path}")

    if agente:
        evaluar_agente(agente, dispositivo, num_episodios=args.episodes)
