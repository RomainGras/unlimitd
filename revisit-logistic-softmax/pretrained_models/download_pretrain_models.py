from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

AutoImageProcessor.save_pretrained("./AutoImageProcessor")
AutoModel.save_pretrained("./AutoModel")