# evolutionary-agent-testbed

- `python -m .venv venv`
- `python -m pip install --upgrade pip`
- `pip install gymnasium minigrid pygame`
- `source .venv/Scripts/activate`

---

```
evolutionary-agent-testbed/
│
├── agents/ quem aprende;
├── algorithms/ como aprende;
├── environment/ onde aprende;
├── configs/
├── utils/
├── assets/
│
├── main.py
├── requirements.txt
└── README.md
```

# Agente

## O que é um agente?

O agente representa um indíviduo da população.

Cada um possui seu próprio DNA (genoma) e será avaliado individualmente.

## Genoma

Representa o DNA do agente. Pode ser a sequência de movimentos (cima, baixo, esquerda ou direita)? Ou pesos de uma rede neural.

## Posição

Guarda as coordenadas do agente no mapa (avaliar necessidade).

## Fitness

Mede o quão proxímo o agente chegou próximo do objetivo.
