import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuração da Página
st.set_page_config(page_title="Estudo de Sistemas Lineares Intervalares", layout="wide")

st.title("Estudo de Equações Lineares Intervalares: Ax = b")
st.markdown("""
Esta ferramenta baseia-se no artigo **"Interval linear systems as a necessary step in fuzzy linear systems"** (Lodwick & Dubois, 2015).
O objetivo é resolver equações do tipo $[a]x = [b]$ determinando o tipo de solução com base na natureza dos intervalos (**Ôntico** ou **Epistêmico**).
""")

# --- BARRA LATERAL (MENU INTERATIVO) ---
st.sidebar.header("Configuração dos Intervalos")

def input_interval(label):
    st.sidebar.subheader(f"Intervalo {label}")
    # Usando sliders e number inputs para flexibilidade
    col1, col2 = st.sidebar.columns(2)
    val_min = col1.number_input(f"Mínimo {label}", value=1.0 if label=='A' else 4.0, step=0.5)
    val_max = col2.number_input(f"Máximo {label}", value=2.0 if label=='A' else 6.0, step=0.5)
    
    if val_min > val_max:
        st.sidebar.error(f"Erro em {label}: Mínimo > Máximo!")
        return None, None
    
    tipo = st.sidebar.radio(
        f"Natureza de {label}", 
        ("Epistêmico (Disjuntivo)", "Ôntico (Conjuntivo)"), 
        index=0 if label=='A' else 1,
        key=f"tipo_{label}",
        help="Epistêmico: Incerteza sobre um valor único. Ôntico: Representa um conjunto real de valores/restrição."
    )
    return (val_min, val_max), tipo

# Inputs
int_a, tipo_a = input_interval("[A]")
int_b, tipo_b = input_interval("[b]")

if int_a is None or int_b is None:
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Legenda de Semântica:**")
st.sidebar.info(
    "- **Epistêmico + Ôntico** -> Caso 1 (Robusto)\n"
    "- **Ôntico + Epistêmico** -> Caso 2 (Controle)\n"
    "- **Ôntico + Ôntico** -> Caso 3 (Clássico)\n"
    "- **Epistêmico + Epistêmico** -> Caso 4 (Unido)"
)

# --- LÓGICA DE SOLUÇÃO (Baseada em Lodwick & Dubois, Seção 3.5) ---

def classificar_caso(tipo_a, tipo_b):
    e = "Epistêmico (Disjuntivo)"
    o = "Ôntico (Conjuntivo)"
    
    if tipo_a == e and tipo_b == o:
        return 1, "Caso 1: Solução Robusta (Tolerance)", r"[A]x \subseteq \mathbf{b}"
    elif tipo_a == o and tipo_b == e:
        return 2, "Caso 2: Solução de Controle (Control)", r"\mathbf{A}x \supseteq [b]"
    elif tipo_a == o and tipo_b == o:
        return 3, "Caso 3: Solução Clássica", r"\mathbf{A}x = \mathbf{b}"
    else:
        return 4, "Caso 4: Solução Unida (United)", r"[A]x \cap [b] \neq \emptyset"

caso_num, caso_nome, caso_latex = classificar_caso(tipo_a, tipo_b)

# Função para resolver 1D ax = b baseada na seção 3.5 do artigo
def resolver_intervalar(a, b, caso):
    a_min, a_max = a
    b_min, b_max = b
    
    # Verificação básica de divisão por zero para simplificação deste demo
    if a_min <= 0 <= a_max:
        return None, "O intervalo [A] contém zero. A solução pode ser ilimitada ou união de intervalos (não tratado neste demo simples)."

    sol_min, sol_max = None, None
    formula = ""

    # Assumindo A > 0 e B > 0 para demonstração clara das fórmulas principais do artigo.
    # O artigo detalha sinais mistos, mas focaremos no caso padrão para visualização.
    if a_min > 0 and b_min >= 0:
        if caso == 1:
            # Caso 1 (Robust): [b_min/a_min, b_max/a_max] SE válido
            # No artigo (pág 12, item 1.a): Omega = [b_min/a_min, b_max/a_max] 
            # *Correção*: O artigo diz b_min <= ax <= b_max para todo a. 
            # Isso implica x >= b_min/a e x <= b_max/a. 
            # O mais restritivo (pior caso) para 'a' em [a_min, a_max] é:
            # x >= b_min / a_min (errado, deve segurar para todo a) -> x >= b_min / a_min 
            # Espera, vamos reler Seção 3.5 item 1(a):
            # "Most constraining case... Omega = [b_min/a_min, b_max/a_max]"
            # A fórmula no texto está: [b_bar/a_bar, b_top/a_top]?? 
            # Não, na seção 3.5 (1.a): [b_min/a_min, b_max/a_max]. 
            # Verificação lógica: se x está na solução, então para a_min, a_min*x >= b_min.
            
            # Vamos usar a lógica de "Inclusão": [a_min*x, a_max*x] subconjunto de [b_min, b_max]
            # a_min*x >= b_min  => x >= b_min/a_min
            # a_max*x <= b_max  => x <= b_max/a_max
            sol_min = b_min / a_min
            sol_max = b_max / a_max
            formula = r"x \in \left[ \frac{\underline{b}}{\underline{a}}, \frac{\overline{b}}{\overline{a}} \right]"
            
            if sol_min > sol_max:
                return None, "Conjunto Solução Vazio (Intervalo Impróprio obtido)."

        elif caso == 2:
            # Caso 2 (Control): [b] subconjunto [A]x
            # [b_min, b_max] subconjunto [a_min*x, a_max*x]
            # a_min*x <= b_min => x <= b_min/a_min
            # a_max*x >= b_max => x >= b_max/a_max
            sol_min = b_max / a_max
            sol_max = b_min / a_min
            formula = r"x \in \left[ \frac{\overline{b}}{\overline{a}}, \frac{\underline{b}}{\underline{a}} \right]"
            
            if sol_min > sol_max:
                return None, "Conjunto Solução Vazio (Não existe x que cubra todo [b])."
        
        elif caso == 3:
            # Caso 3 (Igualdade): Interseção de 1 e 2
            # Muito restritivo. Geralmente vazio a menos que seja um escalar.
            s1_min, s1_max = b_min / a_min, b_max / a_max
            s2_min, s2_max = b_max / a_max, b_min / a_min
            
            # Para não ser vazio, precisamos que os intervalos coincidam (ponto único)
            if np.isclose(s1_min, s2_min) and np.isclose(s1_max, s2_max):
                 sol_min, sol_max = s1_min, s1_max
                 formula = "Solução Pontual (Rara)"
            else:
                 return None, "Conjunto Vazio (A igualdade estrita Ôntica é muito restritiva)."

        elif caso == 4:
            # Caso 4 (United): Aritmética Intervalar Padrão
            # [b_min, b_max] / [a_min, a_max] = [b_min/a_max, b_max/a_min]
            sol_min = b_min / a_max
            sol_max = b_max / a_min
            formula = r"x \in \left[ \frac{\underline{b}}{\overline{a}}, \frac{\overline{b}}{\underline{a}} \right]"

    else:
        # Simplificação para este app didático
        return None, "Para fins didáticos e visualização gráfica, este app foca em A > 0 e b >= 0. O artigo cobre os outros casos de sinais."

    return (sol_min, sol_max), formula

solucao, msg_solucao = resolver_intervalar(int_a, int_b, caso_num)

# --- VISUALIZAÇÃO GRÁFICA ---

col_main, col_expl = st.columns([2, 1])

with col_main:
    st.subheader(f"Análise: {caso_nome}")
    st.latex(caso_latex)
    
    if solucao:
        st.success(f"Solução Encontrada: x ∈ [{solucao[0]:.4f}, {solucao[1]:.4f}]")
        st.latex(msg_solucao)
    else:
        st.error(f"Resultado: {msg_solucao}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Limites do gráfico
    x_limit = 10
    if solucao:
        x_limit = solucao[1] * 1.5
    else:
        x_limit = (int_b[1]/int_a[0]) * 1.5 if int_a[0] > 0 else 10
        
    x_vals = np.linspace(0, x_limit, 400)
    
    # 1. Desenhar o Cone gerado por [A]x
    # Linha inferior: a_min * x
    # Linha superior: a_max * x
    y_min_cone = int_a[0] * x_vals
    y_max_cone = int_a[1] * x_vals
    
    ax.fill_between(x_vals, y_min_cone, y_max_cone, color='skyblue', alpha=0.4, label='Região [A]x')
    ax.plot(x_vals, y_min_cone, color='blue', linestyle='--', linewidth=1)
    ax.plot(x_vals, y_max_cone, color='blue', linestyle='--', linewidth=1)
    
    # 2. Desenhar a Faixa horizontal gerada por [b]
    ax.axhspan(int_b[0], int_b[1], color='orange', alpha=0.4, label='Faixa alvo [b]')
    ax.axhline(int_b[0], color='red', linestyle='-', linewidth=1)
    ax.axhline(int_b[1], color='red', linestyle='-', linewidth=1)
    
    # 3. Desenhar a Solução no Eixo X
    if solucao:
        # Linha grossa verde no eixo X
        ax.plot([solucao[0], solucao[1]], [0, 0], color='green', linewidth=8, solid_capstyle='butt', label='Solução X')
        # Linhas verticais tracejadas para mostrar a interseção
        ax.vlines(solucao[0], 0, max(int_b[1], int_a[1]*solucao[0]), colors='green', linestyles='dotted')
        ax.vlines(solucao[1], 0, max(int_b[1], int_a[1]*solucao[1]), colors='green', linestyles='dotted')
        
        # Anotações
        ax.text(solucao[0], -0.5, f"{solucao[0]:.2f}", ha='center', color='green', fontweight='bold')
        ax.text(solucao[1], -0.5, f"{solucao[1]:.2f}", ha='center', color='green', fontweight='bold')

    ax.set_ylim(-1, int_b[1] * 2)
    ax.set_xlim(0, x_limit)
    ax.set_xlabel("Valor de x")
    ax.set_ylabel("Valor de Ax e b")
    ax.set_title("Visualização Geométrica: Interseção do Cone [A]x com a Faixa [b]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with col_expl:
    st.markdown("### Explicação Teórica")
    
    st.markdown(f"**Interpretação Detectada:** {caso_nome}")
    
    if caso_num == 1:
        st.info("""
        **Contexto:** Procuramos valores de $x$ que permaneçam dentro do alvo **b** (ôntico/rígido) para *qualquer* valor que o parâmetro incerto **A** (epistêmico) possa assumir.
        
        **Operação:**
        Isso exige que o cone azul (todas as possibilidades de A) esteja **totalmente dentro** da faixa laranja na região da solução.
        É a intersecção "pessimista".
        """)
    elif caso_num == 2:
        st.info("""
        **Contexto:** Procuramos $x$ tal que consigamos cobrir *toda* a incerteza do alvo **b** (epistêmico) usando nosso parâmetro de controle **A** (ôntico/faixa de tolerância).
        
        **Operação:**
        Isso exige que a faixa laranja esteja contida dentro do cone azul.
        """)
    elif caso_num == 3:
        st.info("""
        **Contexto:** Igualdade estrita entre dois conjuntos sólidos.
        
        **Operação:**
        Exige que o cone azul e a faixa laranja coincidam perfeitamente. Raramente possível com intervalos com largura.
        """)
    elif caso_num == 4:
        st.info("""
        **Contexto:** Procuramos $x$ tal que *exista* algum valor possível em **A** que resulte em algum valor possível em **b**.
        
        **Operação:**
        É a Aritmética Intervalar Clássica. Basta que o cone azul e a faixa laranja tenham **qualquer interseção**. É a união de todas as possibilidades.
        """)

    st.markdown("---")
    st.markdown("**Detalhes dos Intervalos:**")
    st.write(f"$[A] = [{int_a[0]}, {int_a[1]}]$ ({tipo_a})")
    st.write(f"$[b] = [{int_b[0]}, {int_b[1]}]$ ({tipo_b})")

st.markdown("""
---
*Desenvolvido com base nas definições de: Lodwick, W. A., & Dubois, D. (2015). Interval linear systems as a necessary step in fuzzy linear systems.*
""")
