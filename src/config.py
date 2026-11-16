"""Parâmetros principais e valores padrão usados pelo projeto."""

# CONFIGURAÇÕES PRINCIPAIS
# Ajuste estes parâmetros para controlar custo, duração e tamanho dos testes.

# Número de execuções independentes e sementes para reprodutibilidade.
EXPERIMENT_RUNS = 30
SEED = 42
WORKERS = 8

# Tamanho e duração do experimento.
POPULATION_SIZE = 200
GENERATIONS = 100
MAX_STEPS = 50
GENOME_LENGTH = 50

# Configurações de MLP.
MLP_HIDDEN_SIZES = (8, 16, 24)
# Opções: "relu", "leaky_relu" (alpha fixo em 0.25) e "prelu" (alpha evolutivo).
MLP_ACTIVATION_FUNCTION = "relu"


# HIPERPARÂMETROS AVANÇADOS DOS ALGORITMOS
# Ajuste apenas se quiser alterar o comportamento dos algoritmos.

# Algoritmo genético.
TOURNAMENT_SIZE = 3
SELECTION_AMOUNT = 30
ELITE_SIZE = int(POPULATION_SIZE / 2)
CROSSOVER_RATE = 1.0
MUTATION_RATE = 0.02

# Cross-Entropy Method.
CEM_INITIAL_STD = 1.0
CEM_MIN_STD = 1e-3

# Differential Evolution.
DE_MUTATION_FACTOR_RANGE = (0.0, 1.0)
DE_CROSSOVER_RATE_RANGE = (0.0, 1.0)


# VALORES PADRÃO DO MENU E DA EXECUÇÃO NÃO INTERATIVA
# No terminal interativo, agente, modo, otimizador e ambiente são escolhidos
# pelo menu. Estes valores definem a seleção inicial e servem de fallback.
EXPERIMENT_TYPE = "movement"
EXPERIMENT_MODE = "single"
EXPERIMENT_OPTIMIZER = "genetic"
ENVIRONMENT_TYPE = "frozen_lake"
LOG_LEVEL = "INFO"


# CONSTANTES INTERNAS DO AMBIENTE E DA REPRESENTAÇÃO
# Normalmente não precisam ser alteradas para configurar um experimento.
DEFAULT_MAP_SIZE = "4x4"
RENDER_MODE = None

OBSERVATION_SIZE = 3
BORDER_WIDTH = 1

EMPTY_CELL_VALUE = 0
UNSEEN_CELL_VALUE = -0.25
HOLE_CELL_VALUE = -1
GOAL_CELL_VALUE = 1
BORDER_CELL_VALUE = -0.5
