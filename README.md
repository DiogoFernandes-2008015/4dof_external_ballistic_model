# Simulador de Trajetória - Modelo de Massa Ponto Modificado (MPM)

Este repositório contém uma implementação em Python para a simulação e análise aero-balística de um morteiro de 120mm utilizando o **Modelo de Massa Ponto Modificado (MPM)**. O modelo físico e os coeficientes aerodinâmicos foram baseados estritamente na obra de referência clássica de Robert L. McCoy, *"Modern Exterior Ballistics"*.

O simulador é capaz de modelar trajetórias balísticas convencionais (disparos terrestres) e também cenários de lançamentos horizontais a partir de plataformas móveis em altitude (como estudos paramétricos de liberação de carga útil por aeronaves ou drones).

---

## 📊 Fundamentação Teórica e Física

Diferente de um modelo simplificado de 3 Graus de Liberdade (3DOF), que trata o projétil estritamente como uma massa puntual sob ação do arrasto e da gravidade, este modelo incorpora um quarto grau de liberdade dinâmico essencial para projéteis estabilizados por aletas (não girantes, onde a velocidade de rotação axial $p = 0$): o **Yaw de Repouso** ($\vec{\alpha}_R$).

As equações principais de movimento resolvidas iterativamente para o vetor velocidade $\vec{V}$ são:

$$\frac{d\vec{V}}{dt} = -\frac{\rho S C_D}{2m} v \vec{V} + \frac{\rho S C_{L_\alpha}}{2m} v^2 \vec{\alpha}_R + \vec{g}$$

Onde:
* $\rho$: Densidade do ar (calculada via modelo de atmosfera padrão ISA/ICAO).
* $S$: Área de referência transversal do projétil ($S = \frac{\pi d^2}{4}$).
* $m$: Massa do projétil ($13.585 \text{ kg}$).
* $v$: Magnitude da velocidade relativa ao fluido.
* $\vec{\alpha}_R$: Vetor do *Yaw* (ou *Pitch*) de repouso, determinado pela forma simplificada de Bradley para projéteis aletados:

$$\vec{\alpha}_R = \frac{d}{v^4} \left[ \vec{v} \times (\vec{v} \times \vec{g}) \right]$$

### Aerodinâmica Avançada
Os coeficientes de arrasto ($C_D$) e sustentação ($C_{L_\alpha}$) são atualizados dinamicamente a cada passo de integração em função do Número de Mach e do ângulo de ataque total ($\alpha_t$), utilizando **interpolação polinomial de Lagrange de 4 pontos** a partir das tabelas experimentais do *BRL (Ballistic Research Laboratory)* compiladas no Appendix C do livro do McCoy.

---

## 🛠️ Tecnologias e Dependências

A arquitetura do código foi desenvolvida focando em alto desempenho numérico e precisão de integração:
* **JAX:** Utilizado para infraestrutura de computação de alta performance com suporte a precisão dupla (`float64`).
* **Diffrax:** Biblioteca avançada de equações diferenciais para JAX. O solver escolhido é o `Dopri5` (Runge-Kutta 4/5 adaptativo) com controle de passo via controlador PID.
* **NumPy & Matplotlib:** Para manipulação de matrizes e geração de gráficos de alta resolução.

### Instalação das dependências
```bash
pip install numpy matplotlib jax jaxlib diffrax
````
### 📂 Estrutura do Repositório
O repositório é composto por três scripts principais:

mortar_120mm_mpm.py: O motor físico principal. Integra a trajetória a partir de parâmetros de linha de comando, exibe um relatório detalhado no terminal e salva os gráficos das propriedades do voo.

estudo_h0_vs_angulo.py: Realiza um estudo paramétrico comparando diferentes altitudes de lançamento ($h_0$) versus o ângulo de impacto e alcance final do projétil, simulando um lançamento completamente horizontal ($QE = 0^\circ$).

superficie_v_h0_angulo.py: Gera uma varredura bidimensional em malha variando a velocidade horizontal e a altitude, plotando uma superfície 3D e um mapa de contorno térmico do comportamento do ângulo de impacto.

### 💻 Como Utilizar

### Exemplo 1. Simulação de Trajetória Padrão
Para rodar uma simulação balística individual passando a velocidade de boca (--mv), o ângulo de elevação (--qe) e opcionalmente a altitude inicial (--h0), execute:
Exemplo 1: Disparo convencional ao nível do solo com velocidade de 318 m/s e elevação de 65°
```bash
python mortar_120mm_mpm.py --mv 318 --qe 65 --h0 0
````
# Exemplo 2: Lançamento a partir de uma elevação de 200m com saída customizada
```bash
python mortar_120mm_mpm.py --mv 102 --qe 45 --h0 200 --saida resultados/
````
Gráficos gerados automaticamente:
trajetoria.png:Perfil vertical do voo com mapeamento de cores da velocidade.
angulo_ataque.png: Evolução do ângulo de repouso ($|\alpha_R|$) e Mach ao longo do tempo.
coeficientes_aero.png: Comportamento dinâmico dos coeficientes ($C_D, C_{L_\alpha}, C_{M_\alpha}, C_{M_{qd}}$).
tabelas_coeficientes.png: Visualização das curvas interpoladas vs. dados brutos do livro.

### Exemple 3. Estudo Paramétrico de Altitude ($h_0$) 

Para analisar o impacto da variação de altitude no ângulo final de queda do projétil sob diferentes velocidades de translação horizontal:
````Bash
python estudo_h0_vs_angulo.py
````
Saída: 
h0_vs_angulo_impacto.png (Gráficos comparativos de Alcance e Ângulo).

### 3. Mapeamento Completo de Superfície 3D 
Para gerar uma varredura completa cruzando velocidades de avanço (10 a 40 m/s) e altitudes (20 a 520 metros):
````bash
python superficie_v_h0_angulo.py
````
Saídas:
superficie_3d_angulo.png (Visualização tridimensional),  mapa_contorno_angulo.png (Gráfico de curvas de nível) e os dados brutos salvos em superficie_dados.npz.

📈 Exemplo de Saída do Terminal
Ao executar o script principal, o simulador exibirá uma tabela formatada com o resumo numérico do disparo:

```bash
╔══════════════════════════════════════════════════════════╗
║        RESULTADO — TRAJETÓRIA MPM — MORTEIRO 120mm      ║
╠══════════════════════════════════════════════════════════╣
║  Velocidade de lançamento :   318.00 m/s               ║
║  Ângulo de elevação (QE)  :    65.00 °                 ║
║  Altura de lançamento (h₀):     0.00 m                 ║
╠══════════════════════════════════════════════════════════╣
║  Tempo de voo (ToF)       :    52.41 s                 ║
║  Alcance de impacto       :   6342.7 m                 ║
║  Altitude máxima (Zmax)   :   3125.1 m                 ║
║  Velocidade de impacto    :   215.34 m/s               ║
║  Ângulo de impacto        :    68.12 °                 ║
╚══════════════════════════════════════════════════════════╝
````

📖 Referências
McCoy, Robert L., "Modern Exterior Ballistics: The Launch and Flight Dynamics of Symmetric Projectiles", 2ª Edição. Capítulo 9 (Modified Point Mass Tractable Equations) e Apêndice C (Dados de coeficientes aerodinâmicos para morteiros).
