import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.title("Физика волновых процессов. Задача 5.8")
st.header("Нестеров Дмитрий, ВМК 301 группа")

st.subheader("Условие задачи")

st.markdown(r"""
Исследовать влияние шага дискретизации на погрешность
вычисления спектра пилообразного импульса $y(t)$:
$$y(t) = 
\begin{cases}
    a (1 - \dfrac{t}{\tau}), ~~ 0 \le t \le \tau, \\
    0, ~~ t < 0,~ t > \tau.
\end{cases}$$
Вычислить и построить в зависимости от частоты $\nu$, 
взятой в единицах, пропорциональных $\tau^{-1}$,
дискретный спектр пилообразного импульса длительностью 
$\tau$ при шагах дискретизации $h = 0.05\tau;~0.1\tau;~0.2\tau;~0.4\tau$.
Интервал периодизации ( $T\ge 3\tau$ ) выбрать так, 
чтобы размерность сетки составляла $N=\frac{T}{h}=2^p,~p\in\mathbb{Z}$.
Оценить ширину спектра $\Delta\nu$. Сравнить частоту Найквиста
$\nu_N$ с шириной спектра.
""")

st.subheader("Нахождение непрерывного Фурье-образа")

st.latex(r"""
\dfrac{2\pi}{a}S(\omega)=\int_0^{\tau}(1-\frac{t}{\tau})
e^{-i\omega t}dt=\int_0^{\tau}e^{-i\omega t}dt-\frac{1}{\tau}
\int_0^{\tau}te^{-i\omega t}dt={\text\{по~частям\}}=
\\~\\
=-\frac{1}{i\omega}\left[e^{-i\omega t}\right]\Big|_0^\tau-\frac{1}{\tau}
\left[-\frac{1}{i\omega}t e^{-i\omega t}-\frac{1}{(i\omega)^2}e^{-i\omega t}\right]\Biggr|_0^\tau=
\\~\\
=-\frac{1}{i\omega}\left[e^{-i\omega\tau}-1\right]-\frac{1}{\tau}
\left[-\frac{1}{i\omega}\tau e^{-i\omega\tau}-\frac{1}{(i\omega)^2}e^{-i\omega\tau}+\frac{1}{(i\omega)^2}\right]=
\\~\\
=-\frac{1}{i\omega}e^{-i\omega\tau}+\frac{1}{i\omega}+\frac{1}{i\omega}e^{-i\omega\tau}
+\frac{1}{(i\omega)^2\tau}e^{-i\omega\tau}-\frac{1}{(i\omega)^2\tau}=
\\~\\
=\underbrace{\frac{1}{\omega^2\tau}(1-\cos\omega\tau)}_{\dfrac{2\pi}{a}\large\text{Re}(\omega)}+
i\underbrace{\frac{1}{\omega}\left(\frac{1}{\omega\tau}\sin\omega\tau-1\right)}_{\dfrac{2\pi}{a}\large\text{Im}(\omega)}
\\~\\
G(\omega)=(2\pi)^2|S(\omega)|^2=a^2\left(\frac{1}{\omega^4\tau^2}(1-\cos\omega\tau)^2+
\frac{1}{\omega^2}\left(\frac{1}{\omega\tau}\sin\omega\tau-1\right)^2\right)
\\~\\
\text{После замены } p=\omega\tau:
\\~\\
G(p)=\frac{a^2\tau^2}{p^2}\left[\left(\frac{1-\cos p}{p}\right)^2+\left(\frac{\sin p}{p} - 1\right)^2\right]
""")


st.subheader("Графики Фурье-образа")

a = st.slider("a", min_value=1.0, max_value=10.0, value=4.0, help="Масштабируемость сигнала", key="a1")
tau = st.slider("tau", min_value=1.0, max_value=10.0, value=4.0, help="Длительность сигнала", key="tau1")


def signal(t, a_src, tau_src):
    if t >= tau_src or t <= 0:
        return 0
    return a_src * (1 - t / tau_src)


def re_part(omega, a_src, tau_src):
    return a_src / (2 * np.pi * omega ** 2 * tau_src) * (1 - np.cos(omega * tau_src))


def im_part(omega, a_src, tau_src):
    return a_src / (2 * np.pi * omega) * (1 / (omega * tau_src) * np.sin(omega * tau_src) - 1)


def power_density(omega, a_src, tau_src):
    return (2 * np.pi) ** 2 * (re_part(omega, a_src, tau_src) ** 2 + im_part(omega, a_src, tau_src) ** 2)


x = np.linspace(-10, 10, 1000)
fourier_re = np.vectorize(lambda omega: re_part(omega, a, tau))(x)
fourier_im = np.vectorize(lambda omega: im_part(omega, a, tau))(x)
fourier_abs = np.vectorize(lambda omega: power_density(omega, a, tau))(x)

data = pd.DataFrame({
    "omega": x,
    "Re": fourier_re,
    "Im": fourier_im,
    "sqrt(G)": fourier_abs
})

fig1 = px.line(data, x="omega", y=["Re", "Im"], title="Вещественная и мнимая части")
fig2 = px.line(data, x="omega", y=["sqrt(G)"], title="Спектральная плотность мощности")

st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.markdown(r"""
Полная ширина спектральной плотности мощности:
$$\Delta\omega\sim\frac{14}{\tau}\Rightarrow\nu_{\text{max}}\sim\frac{7}{2\pi\tau}\approx\frac{1.1}{\tau}.$$
""")

st.subheader("Дискретный спектр")

st.latex(r"""
S_h(n)=\frac{1}{N}\sum\limits_{j=0}^{N-1}y(t_j)\text{exp}(-i\frac{2\pi}{N}jn)=
\frac{a}{N}\sum\limits_{j=0}^{N'}(1-j\frac{h}{\tau})\text{exp}(-i\frac{2\pi}{N}jn), \text{ где}\\
N'\text{ -- минимальный номер, при котором } N'\frac{h}{\tau} \ge 1.
""")

st.markdown(r"""
1. $h=0.05\tau$. Пусть $T=64h=3.2\tau>3\tau$.

Тогда $N=64$, частота Найквиста $\nu_N=\frac{1}{2h}=\frac{10}{\tau}$.

Далее делаем подсчеты аналогичным образом:

2. $h=0.1\tau,~T=32h=3.2\tau>3\tau\Rightarrow N=32,~\nu_{N}=\frac{5}{\tau}$

3. $h=0.2\tau,~T=16h=3.2\tau>3\tau\Rightarrow N=16,~\nu_{N}=\frac{2.5}{\tau}$

4. $h=0.4\tau,~T=8h=3.2\tau>3\tau\Rightarrow N=8,~\nu_{N}=\frac{1.25}{\tau}$
""")

st.subheader("Графики для спектральных плотностей мощности")

a_new = st.slider("a", min_value=1.0, max_value=10.0, value=4.0, help="Масштабируемость сигнала", key="a2")
tau_new = st.slider("tau", min_value=1.0, max_value=10.0, value=4.0, help="Длительность сигнала", key="tau2")


def discrete_power(n_dots, h, end_period=True):
    indexes = np.array([j for j in range(n_dots + int(end_period))], dtype=np.float32)
    args = indexes * h
    signal_vectorized = np.vectorize(lambda t: signal(t, a_new, tau_new), otypes=[float])

    def calc_value(n):
        signal_values = signal_vectorized(args)
        exps = np.e ** np.array([complex(0, -2 * np.pi * j * n / n_dots) for j in indexes])
        fourier_values = signal_values * exps
        value = np.sum(fourier_values) / n_dots
        return np.abs(value)

    calc_value_vectorized = np.vectorize(calc_value, otypes=[float])
    power_spectrum = calc_value_vectorized(indexes)

    return args, power_spectrum


pairs = [(64, 0.05 * tau_new), (32, 0.1 * tau_new), (16, 0.2 * tau_new), (8, 0.4 * tau_new)]
for N_src, h_src in pairs:
    dots, values = discrete_power(N_src, h_src)
    fig = px.line(pd.DataFrame({
        "omega": dots,
        "G": values
    }), x="omega", y=["G"], title=f"Дискретизация при N={N_src}, h={h_src}", markers='.')
    st.plotly_chart(fig)
