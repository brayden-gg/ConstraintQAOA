from matplotlib import pyplot as plt
import pandas as pd

plt.rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex' : True,
    "text.latex.preamble": "\\usepackage{bm}"
})


fname = "medium_mixer"
datafile = f"./results/{fname}.csv"
df = pd.read_csv(datafile)
df.set_index("p", inplace=True)
df.groupby("n")["ar"].plot(x="p", y="ar", legend=True)
plt.title("Appromixation Ratio, $r$ vs. Layers of QAOA, $p$", size=20)
plt.xlabel("Layers of QAOA, $p$", size=20)
plt.ylabel("Approximation Ratio", size=20)
plt.tick_params(labelsize=18)

ns = df.n.unique()

plt.legend([f"N={N}" for N in ns], fontsize=16)
# plt.savefig(f"./results/plots/{fname}_ar.png")
plt.show()
plt.close()

df.groupby("n")["pmax"].plot(x="p", y="pmax", legend=True)
plt.title("Probability of $\\bm{z_{\\mathrm{opt}}}$ vs. Layers of QAOA, $p$", size=20)
plt.xlabel("Layers of QAOA, $p$", size=20)
plt.ylabel("Probability of $\\bm{z_{\\mathrm{opt}}}$", size=20)
plt.tick_params(labelsize=18)
ns = df.n.unique()

plt.legend([f"N={N}" for N in ns], fontsize=16)
plt.show()
# plt.savefig(f"./results/plots/{fname}_pmax.png")
plt.close()