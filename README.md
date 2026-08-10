# evolutionary-agent-testbed

- `python -m venv .venv`
- `source .venv/Scripts/activate`
- `python -m pip install --upgrade pip`
- `pip install gymnasium pygame`

---

```
evolutionary-agent-testbed/
│
├── agents/ quem é o indivíduo;
├── evolution/ como a população evolui;
├── environment/ como a população evolui;
├── simulation/ como uma execução é realizada;
├── configs/
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

- Population size: 20 agentes
- Genome length: 20 genes
- Max steps: 20
- Episodes per agent: 1
- Selection: 5 melhores
- Crossover: 15 descendentes
- Mutation rate: 5%
- Generations: 100
