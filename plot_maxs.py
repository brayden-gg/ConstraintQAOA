from matplotlib import pyplot as plt
import json
import numpy as np


gammas = []
betas = []
for i in [1,2,3]:
    folder = f"BILP_penalty_{i}"
    with open(f"./results/{folder}/landscape.json")as file:
        data = json.load(file)



    for point in data:
        gammas += point["gammas"]
        betas += point["betas"]

plt.figure()
# plt.scatter(gammas, betas)
plt.hist(betas, bins=50)
# plt.xlabel("gamma")
plt.ylabel("beta")
plt.xlim([0, np.pi * 2])
plt.ylim([0, np.pi * 2])
plt.title("Optimal values of beta accross examples")
# plt.savefig(f"./results/{folder}/optimal_parameters.png")
plt.show()
plt.close()

# lam = [e["lambda"] for e in data]
# prob_opt = [e["prob_opt"] for e in data]
# plt.figure()
# plt.plot(lam, prob_opt)
# plt.title("Probability of sp.optimize.minimize finding the global minimum (n=100)")
# plt.xlabel("lambda")
# plt.ylabel("probability")
# plt.savefig(f"./results/{folder}/probability_vs_lambda.png")
