# Nsight: olhando a GPU por dentro

Material para a aula. Duas ferramentas da NVIDIA, duas perguntas diferentes:

| | **Nsight Systems** (`nsys`) | **Nsight Compute** (`ncu`) |
|---|---|---|
| Pergunta | Para onde vai o tempo do programa inteiro? | Por que este kernel é rápido ou lento? |
| Como | intercepta as chamadas CUDA, marca tempos, amostra a CPU | roda o kernel várias vezes, lendo alguns contadores de hardware por vez |
| Custo | poucos % | 10–100× ou mais no kernel perfilado |
| Rodar com | o tamanho real do problema | N pequeno, um lançamento só |
| Resultado | linha do tempo | seções por kernel |

Ordem de uso: **nossos próprios cronômetros** (baratos, sempre ligados) → **`nsys`** (confirma os cronômetros, mostra surpresas) → **`ncu`** (só no kernel que domina o tempo).

---

## 1. Como o `nsys` funciona

```bash
nsys profile -o buffon_timeline --stats=true ./buffon -n 1e10
```

1. O `nsys` **inicia o programa** como processo filho e carrega uma biblioteca dentro dele antes do `main`. O código não muda nem é recompilado.
2. **Intercepta as chamadas CUDA** via **CUPTI** (CUDA Profiling Tools Interface). Cada `cudaMalloc`, `cudaMemcpy` e lançamento de kernel ganha um tempo de início e fim. A própria GPU informa quando cada kernel e cada cópia realmente começaram e terminaram.
3. **Amostra a CPU**: interrompe o programa periodicamente e anota qual função estava rodando (perfilamento estatístico).
4. **Rastreia o sistema operacional**: threads, chamadas de sistema (por exemplo `ioctl`, que é como o programa conversa com o driver da GPU).
5. **Grava um relatório** (`.nsys-rep`), que se abre no `nsys-ui` ou no Nsight Systems do Windows. `--stats=true` também imprime tabelas-resumo no terminal.

## 2. Demonstração: Buffon (EP3) no `nsys`

RTX 5070 Laptop (36 SMs, CUDA 13.3), 10¹⁰ agulhas em FP64. Relatório: [`profiling/buffon_timeline.nsys-rep`](profiling/buffon_timeline.nsys-rep).

### 2.1 O nosso cronômetro está certo?

| Medida | Valor |
|---|---|
| `LOOP_TIME` do próprio EP3 (relógio da CPU, em volta do kernel) | 1,4795 s |
| `throw_needles` segundo o `nsys` (relógio da GPU) | **1,4794 s** |

**Sim.** A diferença é de 0,1 ms em 1,48 s. O cronômetro barato do EP3 mede o kernel corretamente.

### 2.2 Onde está a parte serial da lei de Amdahl?

O ajuste do EP3 dava uma parte serial de **a ≈ 0,15 s**. A linha do tempo mostra de onde ela vem:

| Momento | Tempo (ms) |
|---|---|
| primeira chamada CUDA rastreada | 0 |
| primeiro `cudaMalloc` | **154** |
| `throw_needles` começa | 157 |
| `throw_needles` termina | 1 636 |
| última chamada CUDA termina | 1 637 |

- Os **154 ms** entre a primeira chamada CUDA e o primeiro `cudaMalloc` são a **criação do contexto CUDA**: o driver prepara a GPU para o processo. Nesse intervalo há **406 chamadas `ioctl`** ao driver, somando 103 ms.
- **Todo o resto da parte serial é quase nada:**
  - 6 `cudaMalloc`: 1,5 ms no total;
  - 5 `cudaMemcpy`: 0,4 ms;
  - os dois lançamentos de kernel: 0,7 ms;
  - o kernel `mark_sm_ids` (que descobre os IDs dos SMs): 0,8 µs;
  - as cópias são minúsculas: alguns KB.
- **Lição:** a parte serial do Buffon é **o custo fixo de "ligar" a GPU**, não o nosso código. Ela não depende de N, e por isso p → 1 quando N cresce.

### 2.3 O que o perfilador custa

| | sem `nsys` | com `nsys` |
|---|---|---|
| `TIME` (total na GPU) | 1,63 s | 2,06 s |
| `LOOP_TIME` (kernel) | 1,4836 s | 1,4795 s |

O `nsys` acrescenta ~0,4 s no início (a preparação dele mesmo), mas **o kernel não muda**. Por isso o `nsys` serve para medir o programa real.

### 2.4 O que mostrar ao vivo

Abrir `buffon_timeline.nsys-rep` no Nsight Systems e mostrar:

1. a faixa **CUDA API** da thread principal: a criação do contexto, os mallocs, o `cudaDeviceSynchronize` longo (a CPU esperando a GPU);
2. a faixa **CUDA HW**: o kernel `throw_needles` como uma barra única de 1,48 s, e o `mark_sm_ids` quase invisível;
3. o zoom no começo: o "vazio" de 154 ms antes do primeiro malloc = a parte serial de Amdahl.

---

## 3. Como o `ncu` funciona

```bash
ncu --set full -k throw_needles -c 1 -o buffon_kernel ./buffon -n 1e8
```

1. Também inicia o programa e intercepta os lançamentos de kernel. `-k throw_needles -c 1`: só o primeiro lançamento desse kernel.
2. A GPU tem **contadores de hardware** (instruções emitidas, acertos de cache, bytes da DRAM, por que os warps esperam, ...), mas só consegue ler **poucos de cada vez**.
3. Por isso o `ncu` faz **replay do kernel**:
   - salva a memória que o kernel usa;
   - roda o kernel e lê um grupo de contadores;
   - restaura a memória e roda de novo para o próximo grupo;
   - repete **dezenas de vezes**, com os clocks travados para que as passadas sejam comparáveis.
4. Junta as passadas nas seções do relatório:
   - **Speed Of Light**: % do pico de computação e de memória;
   - **Roofline**;
   - **Occupancy**: warps ativos por SM e o que limita;
   - **Warp State Statistics**: por que os warps esperam;
   - **Memory Workload Analysis**;
   - **Source**: contadores por linha (compilar com `-lineinfo`).

**O tempo medido pelo `ncu` não vale como resultado de desempenho**: o `ncu` diz *por que*, não *quanto*.

## 4. Demonstração: Buffon no `ncu`

### 4.1 Primeiro obstáculo: permissão

A primeira tentativa falhou:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

Os contadores de hardware são protegidos por padrão, porque podem revelar o que outros programas fazem na GPU. No WSL, a permissão é dada **no Windows**:

1. abrir o **NVIDIA Control Panel** (ou o NVIDIA App);
2. ativar **Desktop → Enable Developer Settings**, se essa opção ainda estiver desligada;
3. **Developer → Manage GPU Performance Counters → "Allow access to the GPU performance counters to all users"**;
4. aplicar (pode pedir administrador).

Vale mencionar na aula: é o primeiro erro que quase todo mundo encontra com o `ncu`.

### 4.2 Resultados

*(a preencher depois de liberar os contadores)*

O que esperamos ver, para a turma prever antes:

- **Speed Of Light:** computação alta e memória baixíssima. O kernel quase não toca a memória: cada thread acumula num registrador e faz um `atomicAdd` no fim.
- **Gargalo:** as unidades FP64. Nesta GPU, FP64 roda a 1/64 da taxa de FP32 (atributo `FP32_PER_FP64=64`).
- **Ocupação:**
  - 1 bloco de 1024 threads por SM;
  - 58 registradores por thread (dados do `./buffon -i`);
  - o máximo é 1536 threads por SM, então a ocupação teórica é 1024/1536 ≈ 67%.

---

## 5. Comandos

```bash
# linha do tempo, com tabelas no terminal
nsys profile -o nome --stats=true ./programa args

# só as tabelas de um relatório já gravado
nsys stats nome.nsys-rep

# um kernel em detalhe (N pequeno!)
ncu --set full -k nome_do_kernel -c 1 -o nome ./programa args

# resumo do ncu no terminal
ncu --import nome.ncu-rep --page details

# contadores por linha de código: compilar com
nvcc -lineinfo ...
```

## 6. Lições para a aula

1. **Medir antes de otimizar**: o `nsys` confirmou o nosso cronômetro e explicou a parte serial em uma imagem.
2. **A parte serial do Buffon é a criação do contexto CUDA** (~0,15 s): um custo fixo da GPU, não do algoritmo.
3. **Perfilador também custa**: o `nsys` acrescentou 0,4 s ao total, mas não mexeu no kernel. O `ncu` muda muito o tempo, então nunca usamos o tempo dele.
4. **Contadores de hardware são protegidos** (`ERR_NVGPUCTRPERM`): é preciso liberar no driver.
