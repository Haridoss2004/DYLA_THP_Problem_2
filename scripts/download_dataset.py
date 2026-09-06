from datasets import load_dataset

print("Downloading jewellery dataset...")

ds = load_dataset("bzcasper/ai-tool-pool-jewelry-vision")

print("\nDataset downloaded successfully.")

for split, dataset in ds.items():
    print(f"{split}: {len(dataset)} images")

print("\nDataset structure:")
print(ds)