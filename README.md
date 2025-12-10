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

Performance:

Using the trained projection head + OCR fusion:

Dataset Top-1 Accuracy	Top-5 Accuracy
Validation	96.39%	99.16%
Test (harder augmentations)	80.26%	89.36%

These numbers reflect real-world robustness, not memorization (no test images appear in training)

Notes:

The model works even with only 1 real image per card, thanks to synthetic augmentation.

OCR fusion measurably improves accuracy on blurred or partially visible images.

The test split uses heavy augmentation to mimic realistic user photos
