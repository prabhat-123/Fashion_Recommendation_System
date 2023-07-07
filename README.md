# Fashion Recommendation System

This is a Fashion Recommendation System that leverages image similarity search techniques to provide apparel recommendations based on visual preferences. The system applies a preprocessing pipeline, including resizing input images to 299x299 dimensions and extracting 2048-dimensional embeddings using a pre-trained Xception network.

## Features

- Image Similarity Search: The system utilizes image embeddings and cosine similarity to perform efficient similarity matching between fashion apparel items.
- Dimensionality Reduction: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the embeddings to 128, facilitating faster similarity computations.
- Innovative Recommendation Algorithm: The system implements a novel recommendation algorithm that suggests the most similar fashion apparel items based on image similarity.
- User-Friendly Interface: The Fashion Recommendation System is deployed using the Streamlit app, providing users with an intuitive and interactive interface to explore and receive personalized fashion recommendations.

## Requirements

To run the Fashion Recommendation System, ensure that you have the following dependencies installed:

- Python (version 3.7 or higher)
- Streamlit (version 0.88.0 or higher)
- TensorFlow (version 2.6.0 or higher)
- Keras (version 2.6.0 or higher)
- NumPy (version 1.19.5 or higher)
- OpenCV (version 4.5.3 or higher)
- scikit-learn (version 0.24.2 or higher)

## Usage

1. Clone the repository: `git clone https://github.com/your-repo.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Prepare your fashion apparel dataset: Ensure that your dataset is structured appropriately and contains fashion apparel images.
4. Preprocess the dataset: Run the preprocessing pipeline, which involves resizing images and extracting embeddings using the Xception network.
5. Dimensionality Reduction: Apply PCA to reduce the dimensionality of the embeddings to 128.
6. Train the recommendation model: Implement the innovative recommendation algorithm based on image similarity.
7. Deploy the Fashion Recommendation System: Use the Streamlit app to create an interactive interface for users to receive personalized fashion recommendations.

## Examples

To provide a visual representation of the Fashion Recommendation System, below are some screenshots of the Streamlit app:

[Insert screenshots of the app interface here]

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Xception network is utilized as a pre-trained model to extract image embeddings.
- The Streamlit framework is used for deploying the Fashion Recommendation System to production.

## Contact

For questions or inquiries, please contact [srv.ale52@gmail.com].

