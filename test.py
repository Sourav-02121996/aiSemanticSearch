from app.embedding_model import EmbeddingModel

model = EmbeddingModel("all-MiniLM-L6-v2")
sentence = ["I love books"]
print(model.encode(sentence))
