from matplotlib import pyplot as plt
import pandas as pd
import os

plt.rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex' : True,
    "text.latex.preamble": "\\usepackage{bm}"
})

figsize = (12, 9)

fnames = [f for f in os.listdir("./results") if ".csv" in f]



df_min = pd.read_csv("./results/min_mixer.csv")

df_medium = pd.read_csv("./results/medium_mixer.csv")


plt.figure(figsize=figsize)
colors = ["red","orange", "green","blue", "indigo", "violet"]

for i, N in enumerate(df_medium.n.unique()):
    data_medium = df_medium[df_medium.n == N]
    plt.plot(data_medium.p, data_medium.ar, color=colors[i])

for i, N in enumerate(df_min.n.unique()):
    data_min = df_min[df_min.n == N]
    plt.plot(data_min.p, data_min.ar, linestyle="dashed", color=colors[i])

plt.title("Appromixation Ratio, $r$ vs. Layers of QAOA, $p$", size=20)
plt.xlabel("Layers of QAOA, $p$", size=20)
plt.ylabel("Approximation Ratio", size=20)
plt.tick_params(labelsize=18)

ns = df_min.n.unique()
plt.legend([f"N={N}" for N in ns], fontsize=16)


plt.savefig(f"./results/plots/approx_ratio.png")


plt.figure(figsize=figsize)

for i, N in enumerate(df_medium.n.unique()):
    data_medium = df_medium[df_medium.n == N]
    plt.plot(data_medium.p, data_medium.pmax, color=colors[i])

for i, N in enumerate(df_min.p.unique()):
    data_min = df_min[df_min.n == N]
    plt.plot(data_min.p, data_min.pmax, linestyle="dashed", color=colors[i])


plt.title("Probability of $\\bm{z_{\\mathrm{opt}}}$ vs. Layers of QAOA, $p$", size=20)
plt.xlabel("Layers of QAOA, $p$", size=20)
plt.ylabel("Probability of $\\bm{z_{\\mathrm{opt}}}$", size=20)
plt.tick_params(labelsize=18)
ns = df_min.n.unique()

plt.legend([f"N={N}" for N in ns], fontsize=16)
plt.savefig(f"./results/plots/prob_max.png")

# plt.close()

plt.figure(figsize=figsize)

for i, P in enumerate(df_medium.p.unique()):
    data_medium = df_medium[df_medium.p == P]
    plt.plot(data_medium.n, data_medium.depth, color=colors[i])
    # print(data_medium.depth)

for i, P in enumerate(df_min.p.unique()):
    data_min = df_min[df_min.p == P]
    plt.plot(data_min.n, data_min.depth, linestyle="dashed", color=colors[i])


plt.title("Circuit Depth vs. Problem Size (qubits), $N$", size=20)
plt.xlabel("Problem Size (qubits), $N$", size=20)
plt.ylabel("Circuit Depth", size=20)
plt.tick_params(labelsize=18)
ps = df_min.p.unique()

plt.legend([f"p={p}" for p in ps], fontsize=16)
plt.savefig(f"./results/plots/circuit_depth.png")

plt.close()


