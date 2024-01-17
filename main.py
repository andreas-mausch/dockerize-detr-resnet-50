from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection
import click
import colorsys
import itertools
import matplotlib.pyplot as plt
import pyexiv2
import requests
import sys
import torch

def scale_lightness(rgb, scale_l):
  h, l, s = colorsys.rgb_to_hls(rgb[0] / 256.0, rgb[1] / 256.0, rgb[2] / 256.0)
  (r, g, b) = colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
  return (int(r * 256), int(g * 256), int(b * 256))

def plot_results(model, image, results):
  draw = ImageDraw.Draw(image)
  font_file = font_manager.findfont('DejaVu Sans')
  font = ImageFont.truetype(font_file, 52)
  colors = [(255, 0, 0), (0, 128, 255), (255, 255, 0), (99, 0, 255), (0, 255, 0), (255, 99, 0), (255, 0, 255)]

  for score, label, (xmin, ymin, xmax, ymax), color in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist(), itertools.cycle(colors)):
    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=4)

    text = f'{model.config.id2label[label]}: {score:0.2f}'
    text_box = draw.textbbox((xmin, ymax - 10), text, font=font)
    draw.rectangle(text_box, fill=scale_lightness(color, 1.6))
    draw.text((xmin + 2, ymax - 10), text, font=font, fill=(0, 0, 0))
  image.save("detected.jpg", "JPEG")

@click.command(context_settings={'show_default': True})
@click.option('--model', default='facebook/detr-resnet-50', help='The name of the object detection model used')
@click.option('--threshold', default=0.9, type=float, help='Only keep detections with score > threshold')
@click.argument('files', nargs=-1)
def detection(files, model, threshold):
  """Detect objects in the given images.
  FILES are the filenames of the images to generate the descriptions for. They can include wildcards / glob patterns.
  """
  processor = DetrImageProcessor.from_pretrained(model, revision="no_timm")
  model = DetrForObjectDetection.from_pretrained(model, revision="no_timm")

  for argument in files:
    for path in Path.cwd().glob(argument):
      with Image.open(path) as image:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        plot_results(model, image, results)
      print("%s" % (path.relative_to(Path.cwd())))

      for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 0) for i in box.tolist()]
        print("  %s (%d%%) at %s" % (model.config.id2label[label.item()], score.item() * 100, box))

if __name__ == '__main__':
  detection()
