from fastai.vision.all import *

path = Path('training_set')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

# print(dls.show_batch(max_n=6))
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(300)

learn.export()