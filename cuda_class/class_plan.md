# Aula de CUDA: plano

Plano de trabalho da aula de CUDA. Ele muda conforme aprendemos; os conceitos estão reunidos em [`../docs/README.md`](../docs/README.md).

**Idioma: aula e slides em português.**

---

## Contexto

- **Quem dá a aula:** estudante de física da USP, na equipe de CUDA da AGA 511. Cinco anos de experiência dando aulas de inglês: à vontade na lousa e com a turma, iniciante em apresentações com slides.
- **Quem assiste:** a turma da AGA 511 (Métodos Computacionais em Astronomia, IAG/USP, 2º semestre de 2026) e o **Prof. Alex Carciofi**, que trabalha com HPC em astronomia (transporte radiativo por Monte Carlo). A turma já conhece a lei de Amdahl, o método de Monte Carlo e a agulha de Buffon (EP1, EP3); as outras equipes cobrem MPI, OpenMP e OpenACC.
- **Nível:** alto nível, mas preciso. Há um professor de HPC na sala, então toda afirmação precisa ter respaldo na documentação do CUDA ou numa medida feita na nossa própria GPU.
- **O que é:** **uma aula, não uma apresentação.** A lousa carrega o raciocínio (diagramas, deduções, previsões); os slides só carregam o que não dá para desenhar bem (código, gráficos, linhas do tempo, fotos).
- **Nosso material:**
  - EP3: Buffon na GPU, com estudo de Amdahl variando os SMs.
  - EP4: a estrela, em andamento.
  - O hardware: uma RTX 5070 Laptop GPU, 36 SMs, CUDA 13.3.
  - Os profilers: Nsight Systems e Nsight Compute.

## O roteiro (primeiro rascunho)

A ideia geral: **história → fundamentos → modelo de programação → Buffon → a estrela**, com as ferramentas (Nsight) como fio condutor. A ordem e o peso de cada parte vão mudar.

| # | Bloco | Ideia principal | Lousa | Slides / demo |
|---|---|---|---|---|
| 0 | **Pergunta de abertura** | "π com 14 casas decimais: 2,7·10¹¹ anos nesta GPU. Por que tanto tempo, e por que a GPU ainda é 100× mais rápida que a CPU?" | a pergunta | um número |
| 1 | **História** | Das placas de vídeo à computação de uso geral. GPGPU via shaders, depois CUDA (2007), FP64 e caches, depois tensor cores, depois GPUs nos supercomputadores de hoje | uma linha do tempo | fotos das placas, um gráfico do Top500 |
| 2 | **Fundamentos** | Latência vs vazão: a CPU gasta seus transistores em cache e controle, a GPU em muitas ULAs simples. Por que isso combina com Monte Carlo | o esboço do chip CPU vs GPU | - |
| 3 | **O modelo de programação** (Programming Guide) | host/device, kernels, grid → blocos → threads → warps → SMs, SIMT | a hierarquia de threads, desenhada | um kernel mínimo |
| 4 | **Memória** | dois espaços de endereçamento, `cudaMalloc`/`cudaMemcpy`, registradores / shared / global / constant | a hierarquia de memória, desenhada | - |
| 5 | **Trabalhando juntas** | condições de corrida, operações atômicas, a fila de trabalho | a corrida, à mão | - |
| 6 | **Buffon** | tudo isso num programa: threads persistentes, a fila, streams Philox por chunk, seleção de SMs | - | trechos de código |
| 7 | **Medindo Buffon** | Amdahl com n = SMs, p depende de N, FP64 vs FP32 | **a turma prevê antes**, depois mostramos | gráficos do EP3, **nsys / ncu ao vivo** |
| 8 | **A estrela: problemas novos** | uma imagem de 1,5 GB, atômicas em muitos endereços, memória constante, a verificação do RNG entre CPU e GPU | por que cache não ajuda aqui | - |
| 9 | **A estrela: resultados** | o que estiver pronto: o vídeo, os speedups, o perfil | - | o vídeo, gráficos |
| 10 | **Fechamento** | para que a GPU serve, para que não serve, o que aprendemos | as três lições | - |

## Material pronto

- [`nsight.md`](nsight.md): como o `nsys` e o `ncu` funcionam, e o Buffon perfilado (linha do tempo em [`profiling/`](profiling/)). Para o bloco 7, ao vivo. Achado principal: a parte serial do Buffon (~0,15 s) é a criação do contexto CUDA.

## Técnicas de ensino

- **Prever, depois medir.** Antes de cada resultado, a turma chuta (o speedup em 36 SMs? FP32 vs FP64?). Depois, a medida. Mantém a sala pensando, e os nossos dados são reais.
- **Construir na lousa.** Desenhar a hierarquia de threads e os espaços de memória passo a passo, em vez de mostrar um diagrama pronto.
- **Uma ideia por slide.** A maioria dos slides: um gráfico ou um trecho de código, com um título que afirma a conclusão.
- **Ferramentas ao vivo.** Uma linha do tempo curta do `nsys` e a seção Speed Of Light do `ncu` no Buffon, rodadas na frente da turma (com capturas de tela como plano B).
- **Pontos de checagem.** Uma pergunta para a turma no fim de cada bloco.

## Questões em aberto

- [ ] Data e duração da aula (o plano do curso dá cerca de 1 h para as apresentações das equipes).
- [ ] É a aula da equipe ou a individual? Quem mais está na equipe de CUDA, e como o tempo é dividido?
- [x] Idioma: português (aula e slides).
- [ ] Projetor, lousa, dá para rodar demos ao vivo no laptop?
- [ ] Quanto da estrela vai estar pronto até lá?

## Slides

- Ferramenta: a decidir (arquivo PowerPoint, ou um deck feito aqui que exporta para .pptx/PDF).
- Primeiro passo: fechar o roteiro, depois rascunhar slide por slide a partir deste plano.

## Registro

- 2026-11-20: plano criado. Roteiro a partir da primeira ideia: história, fundamentos, Programming Guide, conceitos do Buffon, Buffon, conceitos da estrela, estrela.
- 2026-11-20: idioma definido: português.
- 2026-11-21: Buffon perfilado com `nsys`; `ncu` aguarda liberar os contadores de desempenho no Windows.
