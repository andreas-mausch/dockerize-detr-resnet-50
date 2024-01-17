from transformers import DetrImageProcessor, DetrForObjectDetection

DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
