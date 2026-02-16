import matplotlib.pyplot as plt
from scripts.dataset import PetDataset

# Root-Pfad anpassen falls nötig
root = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"

dataset = PetDataset(root)

# Ein Sample laden
image, mask = dataset[0]

# Tensor -> numpy für matplotlib
image_np = image.permute(1, 2, 0).numpy()
mask_np = mask.squeeze().numpy()

# Plot
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(image_np)
axs[0].set_title("Image")
axs[0].axis("off")

axs[1].imshow(mask_np, cmap="gray")
axs[1].set_title("Mask")
axs[1].axis("off")

axs[2].imshow(image_np)
axs[2].imshow(mask_np, alpha=0.5, cmap="jet")
axs[2].set_title("Overlay")
axs[2].axis("off")

plt.tight_layout()
plt.tight_layout()
plt.savefig("sample_visualization.png")
print("Saved to sample_visualization.png")