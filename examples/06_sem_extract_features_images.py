"""sem_extract_features (images): let a multimodal LLM describe product photos.

What the operator does
    ``DataOp.sem_extract_features(nl_prompt, input_columns, name, output_columns=...)`` also works on images and
    audio. SemPipes treats a string column as images when its values end in .jpg/.jpeg/.png/.bmp/.gif (audio:
    .wav/.mp3/...). Values can be local file paths or URLs. With ``generate_via_code=False`` every image is sent to a
    multimodal LLM (by default Gemini 2.5 Flash), which answers with the requested features.

What this example shows
    Twelve fashion products with photos (``examples/data/fashion``). The LLM sees *only the image*, not the product
    name, and extracts the dominant colour, the product category and the article type. We compare its answers with
    the ground-truth columns of the dataset.

How to run (from the repository root)
    poetry run python examples/06_sem_extract_features_images.py

Runtime / cost
    Under a minute, with 12 multimodal LLM calls (one per image).
"""

from pathlib import Path

import pandas as pd
import skrub

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

DATA_DIR = Path(__file__).parent / "data"

IMAGE_FEATURES = {
    "image_colour": "The dominant colour of the product in one or two words, e.g. 'Navy Blue' or 'Black'.",
    "image_category": "The product category, exactly one of: Apparel, Accessories, Footwear.",
    "image_article": "The type of article in one or two words, e.g. 'Shirts', 'Jeans' or 'Watches'.",
}


def main() -> None:
    styles = pd.read_csv(DATA_DIR / "fashion" / "styles.csv")
    # The CSV stores paths relative to the fashion folder. Absolute paths work from any working directory.
    styles["image_path"] = [str(DATA_DIR / "fashion" / path) for path in styles["image_path"]]

    products = skrub.var("products")
    described = products.sem_extract_features(
        nl_prompt="Describe the fashion product shown in the photo.",
        input_columns=["image_path"],  # only the image: the LLM cannot read the answer off the product name
        name="image_features",
        output_columns=IMAGE_FEATURES,
        generate_via_code=False,  # send each image to the multimodal LLM
    )

    # get_data() returns default values for SemPipes' hidden helper variables. We add our table to them.
    env = described.skb.get_data()
    env["products"] = styles.copy()
    result = described.skb.eval(env)

    print("=== Ground truth vs. what the LLM saw in the image ===")
    columns = ["productDisplayName", "baseColour", "image_colour", "masterCategory", "image_category", "articleType"]
    print(result[columns + ["image_article"]].to_string())

    def agreement(truth: str, predicted: str) -> float:
        return (result[truth].str.lower().str.strip() == result[predicted].str.lower().str.strip()).mean()

    print("\n=== Exact agreement with the ground truth ===")
    print(f"Colour:   {agreement('baseColour', 'image_colour'):.0%}  (e.g. 'Navy Blue' vs 'Blue' counts as wrong)")
    print(f"Category: {agreement('masterCategory', 'image_category'):.0%}")


if __name__ == "__main__":
    main()
