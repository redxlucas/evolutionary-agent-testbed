"""Print the observations and decisions of a seeded MLP in MiniGrid."""

import argparse
import random

import numpy as np

import config
from agents import MLPAgent
from agents.genome.mlp_genome import MLPGenome
from environment import MiniGridEnvironment
from experiment.experiment_types import ActivationFunction


ACTION_NAMES = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
}


def format_grid(observation, agent_index):
    symbols = {
        config.UNSEEN_CELL_VALUE: "?",
        config.EMPTY_CELL_VALUE: ".",
        config.BORDER_CELL_VALUE: "#",
        config.HOLE_CELL_VALUE: "L",
        config.GOAL_CELL_VALUE: "G",
    }
    rows = []
    for row_index, row in enumerate(observation):
        cells = []
        for column_index, value in enumerate(row):
            flat_index = row_index * observation.shape[1] + column_index
            cells.append("A" if flat_index == agent_index else symbols.get(float(value), "?"))
        rows.append(" ".join(cells))
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect each observation entering an MLP in MiniGrid."
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = MiniGridEnvironment()
    genome = MLPGenome.random(
        hidden_size=8,
        output_size=env.ACTION_COUNT,
    )
    agent = MLPAgent.create(
        genome,
        activation=ActivationFunction.RELU,
    )

    try:
        env.reset(seed=args.seed)
        for step in range(args.steps):
            observation = env.get_local_observation()
            agent_index = env.get_agent_observation_index()
            mlp_input = np.delete(observation.flatten(), agent_index)
            outputs = agent.predict(mlp_input)
            action = int(np.argmax(outputs))

            print(f"\nStep {step + 1} | posição={env.get_agent_position()}")
            print("Visão 3x3 (A marca a posição do agente):")
            print(format_grid(observation, agent_index))
            print(f"Observação numérica recebida (3x3):\n{observation}")
            print(f"Entrada do MLP (8 valores, linha a linha sem A):\n{mlp_input}")
            print(f"Saídas do MLP: {outputs}")
            print(f"Ação escolhida: {action} ({ACTION_NAMES[action]})")

            _, reward, terminated, truncated = env.step(action)
            print(
                f"Depois da ação: posição={env.get_agent_position()} | "
                f"recompensa={reward:.2f} | "
                f"terminated={terminated} | truncated={truncated}"
            )
            if terminated or truncated:
                print("Episódio encerrado.")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
