# evolutionary-agent-testbed

- `python -m venv .venv`
- `source .venv/Scripts/activate`
- `python -m pip install --upgrade pip`
- `pip install gymnasium minigrid pygame matplotlib`

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

## Dados do gráfico do modo single

Ao executar `main.py` com `EXPERIMENT_MODE = "single"`, o programa salva um CSV em `experiment_results/`. Cada linha corresponde a uma geração e inclui a média, o desvio padrão e o intervalo de confiança usados no gráfico, além da taxa de sucesso de cada execução individual. O arquivo também registra as configurações e os rótulos do gráfico. Um novo arquivo com data e hora no nome é criado a cada execução.

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
