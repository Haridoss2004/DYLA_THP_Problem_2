# Dataset Audit

## 1. Dataset Overview

Dataset: `bzcasper/ai-tool-pool-jewelry-vision`.
Total images: **5130**. Features are `image` and `label`.

## 2. Split Distribution

| Split | Images |
|---|---:|
| train | 4488 |
| validation | 429 |
| test | 213 |

## 3. Category Distribution

### train

`{"Bracelet": 795, "Earrings": 822, "Necklace": 2196, "Pendant": 345, "Ring": 330}`

### validation

`{"Bracelet": 75, "Earrings": 78, "Necklace": 217, "Pendant": 29, "Ring": 30}`

### test

`{"Bracelet": 32, "Earrings": 48, "Necklace": 100, "Pendant": 20, "Ring": 13}`

## 4. Image Resolution Statistics

- **train:** `{"1000x1000": 6, "1240x1860": 3, "1535x2048": 330, "1536x2048": 735, "1600x1600": 801, "183x275": 3, "187x270": 3, "189x267": 6, "194x259": 6, "197x255": 3, "198x254": 3, "199x253": 3, "2000x2000": 3, "2048x1536": 15, "2048x2048": 663, "206x300": 3, "207x243": 3, "213x237": 3, "224x224": 264, "225x225": 75, "235x215": 3, "242x209": 3, "250x250": 3, "252x200": 3, "258x195": 3, "259x194": 6, "262x262": 1347, "264x191": 3, "275x183": 3, "307x400": 3, "350x350": 3, "395x395": 3, "400x400": 6, "440x440": 12, "450x682": 3, "463x387": 3, "463x527": 3, "488x361": 3, "488x760": 3, "500x500": 3, "550x550": 3, "594x300": 3, "600x600": 3, "604x728": 3, "612x586": 3, "613x789": 3, "622x512": 3, "628x360": 3, "629x538": 3, "629x667": 3, "640x640": 3, "646x648": 3, "654x744": 3, "673x428": 3, "675x654": 3, "678x667": 3, "680x680": 3, "684x595": 3, "686x673": 3, "690x520": 3, "690x590": 3, "690x611": 3, "690x612": 3, "690x658": 3, "690x664": 3, "690x668": 3, "690x675": 9, "690x679": 3, "690x681": 6, "690x683": 3, "690x688": 3, "690x692": 3, "690x812": 3, "720x1440": 6, "720x659": 3, "750x500": 6, "760x500": 3, "800x768": 3, "800x800": 9}`
- **validation:** `{"1000x1000": 1, "1000x548": 1, "1080x1440": 1, "1500x1500": 1, "1535x2048": 39, "1536x2048": 58, "1600x1600": 81, "194x259": 1, "2048x1536": 1, "2048x2048": 59, "204x247": 2, "207x244": 1, "208x243": 1, "211x239": 1, "224x224": 24, "224x225": 1, "225x225": 7, "262x262": 134, "275x183": 1, "320x400": 1, "350x350": 1, "400x400": 1, "440x440": 1, "456x480": 1, "502x610": 1, "690x443": 1, "690x675": 1, "690x677": 2, "690x678": 1, "690x681": 1, "760x500": 1, "800x800": 1}`
- **test:** `{"1024x1024": 1, "1200x800": 1, "1280x1645": 1, "1535x2048": 7, "1536x2048": 34, "1600x1600": 43, "189x267": 1, "2000x2000": 1, "2048x1536": 1, "2048x2048": 21, "224x224": 17, "225x225": 2, "245x205": 1, "262x262": 70, "350x350": 1, "360x360": 1, "414x414": 1, "600x600": 1, "640x640": 1, "645x690": 1, "667x681": 1, "690x465": 1, "768x768": 1, "870x1110": 1, "900x900": 1, "941x934": 1}`

Representative decoded images were inspected by category; deterministic filenames, dimensions, and formats are recorded in `dataset_audit.json`.

## 5. Exact Duplicate Analysis

- Unique SHA-256 hashes: **5130**.
- Exact duplicate groups: **0**; images in groups: **0**.
- Cross-split exact duplicate groups: **0**.

## 6. Perceptual Duplicate Analysis

The audit uses a dependency-free 64-bit average perceptual hash as a reproducible screening equivalent; it is not proof of product identity.
- Unique perceptual hashes: **4849**.
- Same-hash groups: **104**.
- Near-duplicate pairs at Hamming distance <= 4: **18859**.

## 7. Cross-Split Leakage Analysis

- Cross-split same perceptual-hash groups: **38**.
- Cross-split near-duplicate pairs: **3715**.
- Cross-split filename source-stem groups: **0**.

These findings mean raw split retrieval results must be treated as potentially optimistic.

## 8. Product Identity Analysis

- Verified product ID: **No**.
- Non-image columns: `label`.
The dataset exposes only image and label. Filenames and row indices can support stable image-level IDs, but no verified product, SKU, or manufacturer identity exists.

## 9. Dataset Limitations

- Multiple catalogue rows may be views, augmentations, or images of the same physical item.
- Class labels identify jewellery categories, not products.
- Filename patterns are source clues, not validated identity metadata.

## 10. Recommended Retrieval Benchmark

Use generated image-level IDs only, explicitly named as image identities. Build the catalogue from one leakage-screened partition, hold out query images whose exact/perceptual/source-stem groups overlap the catalogue, and report results separately for clean versus suspicious groups. For product-level claims, collect real phone queries with manually verified item provenance; do not use the raw labels as product IDs.

## 11. Risks and Mitigations

- Raw train/validation/test retrieval metrics are at risk of leakage when exact or visually near-identical images cross splits.
- Filename source-stem matches are heuristic evidence only and cannot prove physical-product identity.
- A query that is itself present in the catalogue would measure lookup, not generalization to a new photograph.
