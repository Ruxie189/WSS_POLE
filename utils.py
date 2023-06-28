import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


#################################################################### ASSETS ####################################################################
################################################################################################################################################
word_dict = {'aeroplane': ['aeroplane', 'Aircraft', 'Airplane', 'Plane', 'Jet', 'Helicopter'],
             'bicycle': ['bicycle', 'Bike', 'Cycle', 'Pedal bike', 'Two-wheeler', 'Velocipede'],
             'bird': ['bird', 'Avian', 'Fowl', 'Feathered friend', 'Winged creature', 'Songbird'],
             'boat': ['boat', 'Ship', 'Vessel', 'Watercraft', 'Canoe', 'Yacht'],
             'bottle': ['bottle', 'Flask', 'Container', 'Jar', 'Vial', 'Canteen'],
             'bus': ['bus', 'Coach', 'Transit', 'Omnibus', 'Motorbus', 'School bus'],
             'car': ['car', 'Automobile', 'Vehicle', 'Sedan', 'Carriage', 'Coupe'],
             'cat': ['cat', 'Feline', 'Kitty', 'Tomcat', 'Tabby', 'Puss'],
             'chair': ['chair', 'Seat', 'Armchair', 'Recliner', 'Stool', 'Throne'],
             'cow': ['cow', 'Bovine', 'Heifer', 'Bull', 'Ox', 'Cattle'],
             'dining table': ['dining table', 'Kitchen table', 'Dinner table', 'Breakfast table', 'Banquet table', 'Trestle table'],
             'dog': ['dog', 'Canine', 'Puppy', 'Hound', 'Pooch', 'Fido'],
             'horse': ['horse', 'Equine', 'Mare', 'Stallion', 'Pony', 'Gelding'],
             'motorbike': ['motorbike', 'Motorcycle', 'Bike', 'Scooter', 'Motor scooter', 'Moped'],
             'player': ['person', 'individual', 'Human', 'People', 'Citizen', 'Human being'],
             'potted plant': ['potted plant', 'Houseplant', 'Flowerpot', 'Planter', 'Indoor plant', 'Pot plant'],
             'sheep': ['sheep', 'Lamb', 'Ewe', 'Ram', 'Flock', 'Woolly'],
             'sofa': ['sofa', 'Couch', 'Loveseat', 'Settee', 'Sectional', 'Chesterfield'],
             'train': ['train', 'Railway', 'Locomotive', 'Subway', 'Monorail', 'Tram'],
             'tv monitor': ['tv monitor', 'Television', 'Display screen', 'Flat screen', 'Computer monitor', 'Video display']}

gpt_prompts = {'aeroplane': ['aeroplane', 'aircraft', 'airplane', 'plane'],
               'bicycle': ['bicycle', 'bike', 'cycle', 'pedal bike'],
               'bird': ['bird', 'avian', 'fowl', 'feathered friend'],
               'boat': ['boat', 'ship', 'vessel', 'watercraft'],
               'bottle': ['bottle', 'flask', 'container', 'jar'],
               'bus': ['bus', 'coach', 'transit', 'omnibus'],
               'car': ['car', 'automobile', 'vehicle', 'sedan'],
               'cat': ['cat', 'feline', 'kitty', 'tomcat'],
               'chair': ['chair', 'seat', 'armchair', 'recliner'],
               'cow': ['cow', 'bovine', 'heifer', 'bull'],
               'dining table': ['dining table', 'kitchen table', 'dinner table', 'breakfast table'],
               'dog': ['dog', 'canine', 'puppy', 'hound'],
               'horse': ['horse', 'equine', 'mare', 'stallion'],
               'motorbike': ['motorbike', 'motorcycle', 'bike', 'scooter'],
               'player': ['person', 'individual', 'human', 'people'],  # player
               'potted plant': ['potted plant', 'houseplant', 'flowerpot', 'planter'],
               'sheep': ['sheep', 'lamb', 'ewe', 'ram'],
               'sofa': ['sofa', 'couch', 'loveseat', 'settee'],
               'train': ['train', 'locomotive', 'subway', 'monorail'],
               'tv monitor': ['tv monitor', 'television', 'display screen', 'flat screen']}

bnc_prompts = {'aeroplane': ['aeroplane', 'biplane', 'aircraft', 'microlight', 'airplane', 'airliner', 'airship', 'plane', 'monoplane', 'glider', 'seaplane', 'jetliner'], 'bicycle': ['bicycle', 'bike', 'handlebar', 'tricycle'],
               'bird': ['bird', 'finch', 'skylark', 'curlew'], 'boat': ['boat', 'moor', 'yacht', 'sail'],
               'bottle': ['lemonade', 'jar', 'brandy'], 'bus': ['bus', 'train', 'tram'],
               'car': ['car', 'vehicle', 'Bmw', 'van'], 'cat': ['cat', 'dog', 'kitten', 'tabby'],
               'chair': ['chair', 'sofa', 'sit', 'upholstered'], 'cow': ['cow', 'calve', 'sheep', 'milking'],
               'dining table': ['dining table', 'glass-topped', 'marble-topped', 'sideboard'], 'dog': ['dog', 'cat', 'puppy', 'alsatian'],
               'horse': ['horse', 'stallion', 'ride', 'mare'], 'motorbike': ['bicycle', 'sidecar', 'motorbike'],
               'player': ['player', 'person', 'someone', 'individual'], 'potted plant': ['potted plant', 'submerged oxygenating', 'pot-grown', 'non-flowering'],
               'sheep': ['sheep', 'cattle', 'cow', 'goat'], 'sofa': ['sofa', 'settee', 'armchair', 'coverlet'],
               'train': ['train', 'passenger', 'Intercity', 'railway'], 'tv monitor': ['tv monitor', 'low-radiation', 'svga', '14-inch']}

gn_prompts = {"aeroplane": ["aeroplane", "Tunng", "Martin Solveig", "Alessi Ark"], "bicycle": ["bicycle", "bike", "scooter", "bicycles"], "bird": ["bird", "birds", "raptor", "owl"], "boat": ["boat", "boats", "sailboat", "motorboat"], "bottle": ["bottle", "bottles", "jug", "corked bottle"], "bus": ["bus", "buses", "Bus", "busses"], "car": ["car", "vehicle", "cars", "SUV"], "cat": ["cat", "cats", "dog", "kitten"], "chair": ["chair", "chairs", "Chair", "chairperson"], "cow": ["cow", "cows", "pig", "dairy cow"], "dining table": [
    "dining table", "tables", "tray", "dining room"], "dog": ["dog", "dogs", "puppy", "pitbull"], "horse": ["horse", "horses", "racehorse", "stallion"], "motorbike": ["motorbike", "bicycle", "bikes", "mountain bike"], "player": ["person", "someone", "persons", "woman"], "potted plant": ["potted plant", "plants", "Plant", "factory"], "sheep": ["sheep", "lambs", "cows", "goats"], "sofa": ["sofa", "couch", "settee", "sofas"], "train": ["train", "trains", "Train", "commuter train"], "tv monitor": ["tv monitor", "monitoring", "monitors", "monitored"]}

ew_prompts = {"aeroplane": ["aeroplane", "airplane", "aircraft", "biplane"], "bicycle": ["bicycle", "bicycle", "bike", "bike"], "bird": ["bird", "birds", "reptile", "waterfowl"], "boat": ["boat", "boat", "motorboat", "yacht"], "bottle": ["bottle", "bottles", "flask", "jug"], "bus": ["bus", "buse", "bus", "bus"], "car": ["car", "vehicle", "truck", "automobile"], "cat": ["cat", "dog", "cat", "kitten"], "chair": ["chair", "chair", "chairman", "chairperson"], "cow": ["cow", "goat", "pig", "cow"], "dining table": [
    "dining table", "tables", "table", "below"], "dog": ["dog", "puppy", "cat", "pet"], "horse": ["horse", "hors", "horse", "horse"], "motorbike": ["motorbike", "bicycle", "motorbike", "motorcycle"], "player": ["player", "persons", "individual", "person"], "potted plant": ["potted plant", "plant", "flower", "herbaceous"], "sheep": ["sheep", "cattle", "goat", "sheep"], "sofa": ["sofa", "couch", "cupboard", "bathtub"], "train": ["train", "intercity", "freight", "train"], "tv monitor": ["tv monitor", "monitor", "monitoring", "monitore"]}

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "player",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]

################################################################################################################################################
################################################################################################################################################

#################################################################### FUNCTIONS #################################################################
################################################################################################################################################


class TorchvisionNormalize:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):

        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)
        proc_img[..., 0] = (imgarr[..., 0] / 255.0 -
                            self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255.0 -
                            self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255.0 -
                            self.mean[2]) / self.std[2]

        return proc_img


def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones(
            (cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def pre_process(img):
    normalize = TorchvisionNormalize()
    img = center_crop(img, 512)
    img = normalize(img)
    img = HWC_to_CHW(img)
    return img


def to_torch(img):
    img = torch.from_numpy(img)
    return img


def get_similarity(model, image_input, text_inputs):

    # Calculate features
    text = text_inputs
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = torch.cat(
            [model.encode_text(text_input) for text_input in text])

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity
################################################################################################################################################
################################################################################################################################################


###############################################################  PATHS  ########################################################################
################################################################################################################################################

dataset_path = osp.join("assets", "JPEG_IMAGES")
cam_mask_path = osp.join("assets", "CAM_MASKS")
vis_path = osp.join("assets", "COMBINED")
LOD = [gpt_prompts, bnc_prompts, gn_prompts, ew_prompts]
LOD_names = ["gpt", "bnc", "gn", "ew"]


################################################################################################################################################
################################################################################################################################################


with open(osp.join("assets", "GPT_v2_4.json"), "rb") as f:
    prompt_dict = pickle.load(f)
