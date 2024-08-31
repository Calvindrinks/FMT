# %%
import torch
import os

torch.set_printoptions(precision=3, sci_mode=False)
for i in range(20):
    checkpoint = i * 50
    policy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())),
        "logs",
        "solo9",
        "e12",
        "model_" + str(checkpoint) + ".pt",
    )
    policy = torch.load(policy_path)
# print(policy["model_state_dict"].keys())
# print(policy["infos"])
    a = policy["model_state_dict"]["embedding_layer.weight"]
    print(f"checkpoint: {checkpoint} weights: {a}")

# %%
