# evolutionary-agent-testbed

Projeto em Python para executar experimentos com agentes evolutivos nos ambientes Frozen Lake e MiniGrid. A execução é feita pelo terminal; ao final, o programa gera arquivos de resultados e gráficos.

## Requisitos

- Python 3.13 ou superior.
- Git para clonar o repositório.
- Acesso à internet para instalar o projeto e suas dependências.

## Instalação e execução local

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/redxlucas/evolutionary-agent-testbed.git
cd evolutionary-agent-testbed
```

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe main.py
```

### macOS ou Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
PYTHONPATH=src python main.py
```

Execute a partir da raiz do repositório. O `PYTHONPATH` aponta para `src`, onde estão os módulos da aplicação. Inicie o programa em um terminal interativo para abrir o menu de navegação.

## Configurar e executar um experimento

No menu, use as setas para cima e para baixo e pressione Enter para selecionar. Esc volta ou encerra o menu.

1. Escolha **Configurar e executar experimento**.
2. Selecione o agente: Movement, ADALINE ou MLP.
3. Selecione o modo, o otimizador disponível e o ambiente: Frozen Lake ou MiniGrid.
4. Aguarde a execução e o gráfico. Ctrl+C interrompe o experimento atual.

Os modos disponíveis são execução única, comparação de estratégias de fitness e comparação de configurações MLP. A comparação de configurações está disponível para MLP. Movement e ADALINE usam algoritmo genético; MLP também pode usar Cross-Entropy ou Differential Evolution.

O menu escolhe agente, modo, otimizador e ambiente. Para ajustar o custo e a duração, edite `src/config.py` antes de iniciar. Os parâmetros principais são:

- `EXPERIMENT_RUNS`: número de execuções independentes.
- `SEED`: semente base para reprodutibilidade.
- `WORKERS`: processos paralelos.
- `POPULATION_SIZE`: indivíduos por geração ou iteração.
- `GENERATIONS`: gerações do algoritmo genético ou iterações dos otimizadores.
- `MAX_STEPS`: limite de passos por episódio.
- `MLP_HIDDEN_SIZES`: tamanhos de camada usados nas comparações de configuração.

Os valores padrão são 30 execuções, população de 200 indivíduos, 100 gerações/iterações e 50 passos por episódio. A duração depende do computador e da combinação escolhida.

## Onde encontrar os resultados

- `experiment_results/`: contém os CSVs dos experimentos utilizados na monografia, com as métricas dos gráficos e os parâmetros de cada execução. Os arquivos dessa pasta também podem ser usados para recriar os gráficos compatíveis pelo menu.
- `experiment_results/plots/`: gráficos PNG gerados automaticamente. O menu **Gerar gráfico de resultados salvos** permite selecionar um CSV compatível e criar novamente o gráfico.
- `experiment_diagnostics/`: CSVs com detalhes por execução, como sucessos e falhas observados durante os experimentos.
- `logs/`: registros de depuração do processo principal e dos workers. A pasta é criada automaticamente.

Os diretórios de resultados são criados quando necessários. Arquivos com o mesmo nome de log são sobrescritos na execução seguinte.
