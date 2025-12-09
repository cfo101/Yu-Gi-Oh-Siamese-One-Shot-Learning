# Yu-Gi-Oh-Siamese-One-Shot-Learning
This project builds a one-shot image recognition system that identifies Yu-Gi-Oh! cards using a combination of:

- OpenCLIP ViT-B/16 visual embeddings

- A trained Siamese projection head using triplet loss + hard negative mining

- OCR fusion (PaddleOCR) to assist when card text is visible

- A realistic evaluation split using synthetic validation/test augmentations

The goal: reliably identify a card even if only one real image of that card exists.

Key Features:

- One-shot learning setup — Vast majority of cards only have one image to learn from

- Triplet-trained projection head improves separation between cards

- Hard-negative mining dynamically selects challenging negatives

- OCR text fusion maps detected text to real card names to boost accuracy

- Augmented evaluation split tests the model against harsher images than easy 100% clean ones
