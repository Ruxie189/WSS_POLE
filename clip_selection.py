import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt
import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse
import pickle as pkl

if __name__ == '__main__':

    # seed_torch(seed=1)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

parser.add_argument("--corpus", default='British', type=str)
args = parser.parse_args()

def NumpyToPil(img):
    return Image.fromarray(img)
voc = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
       'chair', 'cow', 'dining table','dog','horse', 'motorbike', 'player',
       'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

coco = ['player', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']

if args.corpus == 'British':
    print("British National Corpus")
    prompts = {'aeroplane':['aeroplane','biplane', 'aircraft','microlight'], 'bicycle': ['bicycle','bike','handlebar','tricycle'], 
            'bird': ['bird','finch', 'skylark', 'curlew'], 'boat': ['boat','moor','yacht','sail'], 
            'bottle': ['lemonade','jar','brandy'], 'bus': ['bus','train','tram'], 
            'car' : ['car','vehicle','Bmw','van'], 'cat': ['cat','dog', 'kitten','tabby'], 
            'chair': ['chair','sofa','sit','upholstered'], 'cow': ['cow','calve','sheep','milking'], 
            'dining table': ['dining table','glass-topped','marble-topped','sideboard'],'dog' : ['dog','cat','puppy','alsatian'],
            'horse': ['horse','stallion','ride','mare'], 'motorbike': ['bicycle','sidecar','motorbike'], 
            'player': ['player','person','someone','individual'], 'potted plant': ['potted plant','submerged oxygenating','pot-grown','non-flowering'], 
            'sheep': ['sheep','cattle','cow','goat'], 'sofa': ['sofa','settee','armchair', 'coverlet'], 
            'train': ['train','passenger','Intercity','railway'], 'tv monitor': ['tv monitor','low-radiation','svga','14-inch']}
elif args.corpus == 'British_2ch':
    print("British National Corpus")
    prompts = {'aeroplane':['aeroplane','biplane'], 'bicycle': ['bicycle','bike'], 
            'bird': ['bird','finch'], 'boat': ['boat','moor'], 
            'bottle': ['bottle','lemonade'], 'bus': ['bus','train'], 
            'car' : ['car','vehicle'], 'cat': ['cat','dog'], 
            'chair': ['chair','sofa'], 'cow': ['cow','calve'], 
            'dining table': ['dining table','glass-topped'],'dog' : ['dog','cat'],
            'horse': ['horse','stallion'], 'motorbike': ['motorbike','bike'], 
            'player': ['player','person'], 'potted plant': ['potted plant','submerged oxygenating'], 
            'sheep': ['sheep','cattle'], 'sofa': ['sofa','settee'], 
            'train': ['train','passenger'], 'tv monitor': ['tv monitor','low-radiation']}
elif args.corpus == 'British_3ch':
    print("British National Corpus")
    prompts = {'aeroplane':['aeroplane','biplane', 'aircraft'], 'bicycle': ['bicycle','bike','handlebar'], 
            'bird': ['bird','finch', 'skylark'], 'boat': ['boat','moor','yacht'], 
            'bottle': ['bottle','lemonade','jar'], 'bus': ['bus','train','tram'], 
            'car' : ['car','vehicle','Bmw'], 'cat': ['cat','dog', 'kitten'], 
            'chair': ['chair','sofa','sit'], 'cow': ['cow','calve','sheep'], 
            'dining table': ['dining table','glass-topped','marble-topped'],'dog' : ['dog','cat','puppy'],
            'horse': ['horse','stallion','mare'], 'motorbike': ['motorbike','bike','motorcycle'], 
            'player': ['player','person','someone'], 'potted plant': ['potted plant','submerged oxygenating','pot-grown'], 
            'sheep': ['sheep','cattle','cow'], 'sofa': ['sofa','settee','armchair'], 
            'train': ['train','passenger','Intercity'], 'tv monitor': ['tv monitor','low-radiation','svga']}
elif args.corpus == 'British_5ch':
    print("British National Corpus")
    prompts = {'aeroplane':['aeroplane','biplane', 'aircraft','microlight','flying'], 'bicycle': ['bicycle','bike','handlebar','tricycle','scooter'], 
            'bird': ['bird','finch', 'skylark', 'curlew','birds'], 'boat': ['boat','moor','yacht','sail','ship'], 
            'bottle': ['bottle','lemonade','jar','brandy','swig'], 'bus': ['bus','train','tram','taxi','passenger'], 
            'car' : ['car','vehicle','Bmw','van','mercedes'], 'cat': ['cat','dog', 'kitten','tabby','feline'], 
            'chair': ['chair','sofa','sit','upholstered','armchair'], 'cow': ['cow','calve','sheep','milking','cattle'], 
            'dining table': ['dining table','glass-topped','marble-topped','sideboard','sidetable'],'dog' : ['dog','cat','puppy','alsatian','mongrel'],
            'horse': ['horse','stallion','pony','mare','gelding'], 'motorbike': ['motorbike','bike','motorcycle','ford escort','moped'], 
            'player': ['player','person','persons','someone','individual'], 'potted plant': ['potted plant','submerged oxygenating','pot-grown','non-flowering','half-hardy'], 
            'sheep': ['sheep','cattle','cow','goat','ewe'], 'sofa': ['sofa','settee','armchair', 'coverlet','bedspread'], 
            'train': ['train','passenger','Intercity','paddington','rail'], 'tv monitor': ['tv monitor','low-radiation','svga','14-inch','supervga']}
elif args.corpus == 'Google':
    print("Google News")
    prompts = {"aeroplane": ["aeroplane", "Tunng", "Martin Solveig", "Alessi Ark"], 
    "bicycle": ["bicycle", "bike", "scooter", "bicycles"], 
    "bird": ["bird", "birds", "raptor", "owl"], "boat": ["boat", "boats", "sailboat", "motorboat"], 
    "bottle": ["bottle", "bottles", "jug", "corked bottle"], "bus": ["bus", "buses", "Bus", "busses"], 
    "car": ["car", "vehicle", "cars", "SUV"], "cat": ["cat", "cats", "dog", "kitten"], 
    "chair": ["chair", "chairs", "Chair", "chairperson"], 
    "cow": ["cow", "cows", "pig", "dairy cow"], 
    "dining table": ["dining table", "tables", "tray", "dining room"], 
    "dog": ["dog", "dogs", "puppy", "pitbull"], "horse": ["horse", "horses", "racehorse", "stallion"], 
    "motorbike": ["motorbike", "bicycle", "bikes", "mountain bike"], "player": ["person", "someone", "persons", "woman"], 
    "potted plant": ["potted plant", "plants", "Plant", "factory"], "sheep": ["sheep", "lambs", "cows", "goats"], 
    "sofa": ["sofa", "couch", "settee", "sofas"], "train": ["train", "trains", "Train", "commuter train"], 
    "tv monitor": ["tv monitor", "monitoring", "monitors", "monitored"]}
elif args.corpus == 'Google_2ch':
    print("Google News")
    prompts = {"aeroplane": ["aeroplane", "Tunng"],
    "bicycle": ["bicycle", "bike"], 
    "bird": ["bird", "birds"], "boat": ["boat", "boats"], 
    "bottle": ["bottle", "bottles"], "bus": ["bus", "buses"], 
    "car": ["car", "vehicle"], "cat": ["cat", "cats"], 
    "chair": ["chair", "chairs"], 
    "cow": ["cow", "cows"], 
    "dining table": ["dining table", "tables"], 
    "dog": ["dog", "dogs"], "horse": ["horse", "horses"], 
    "motorbike": ["motorbike", "bicycle"], "player": ["person", "someone"], 
    "potted plant": ["potted plant", "plants"], "sheep": ["sheep", "lambs"], 
    "sofa": ["sofa", "couch"], "train": ["train", "trains"], 
    "tv monitor": ["tv monitor", "monitoring"]}
elif args.corpus == 'Google_3ch':
    print("Google News")
    prompts = {"aeroplane": ["aeroplane", "Tunng", "Martin Solveig"], 
    "bicycle": ["bicycle", "bike", "scooter"], 
    "bird": ["bird", "birds", "raptor"], "boat": ["boat", "boats", "sailboat"], 
    "bottle": ["bottle", "bottles", "jug"], "bus": ["bus", "buses", "Bus"], 
    "car": ["car", "vehicle", "cars"], "cat": ["cat", "cats", "dog"], 
    "chair": ["chair", "chairs", "Chair"], 
    "cow": ["cow", "cows", "pig"], 
    "dining table": ["dining table", "tables", "tray"], 
    "dog": ["dog", "dogs", "puppy"], "horse": ["horse", "horses", "racehorse"], 
    "motorbike": ["motorbike", "bicycle", "bikes"], "player": ["person", "someone", "persons"], 
    "potted plant": ["potted plant", "plants", "Plant"], "sheep": ["sheep", "lambs", "cows"], 
    "sofa": ["sofa", "couch", "settee"], "train": ["train", "trains", "Train"], 
    "tv monitor": ["tv monitor", "monitoring", "monitors"]}
elif args.corpus == 'Google_5ch':
    print("Google News")
    prompts = {"aeroplane": ["aeroplane", "Tunng", "Martin Solveig", "Alessi Ark","Lemon Jelly"], 
    "bicycle": ["bicycle", "bike", "scooter", "bicycles","motorcycle"], 
    "bird": ["bird", "birds", "raptor", "owl","squirrel"], "boat": ["boat", "boats", "sailboat", "motorboat","fishing boat"], 
    "bottle": ["bottle", "bottles", "jug", "corked bottle","carafe"], "bus": ["bus", "buses", "Bus", "busses","minibus"], 
    "car": ["car", "vehicle", "cars", "SUV","minivan"], "cat": ["cat", "cats", "dog", "kitten","feline"], 
    "chair": ["chair", "chairs", "Chair", "chairperson","chairwoman"], 
    "cow": ["cow", "cows", "pig", "dairy cow","bovines"], 
    "dining table": ["dining table", "tables", "tray", "dining room","banquette"], 
    "dog": ["dog", "dogs", "puppy", "pitbull","pooch"], "horse": ["horse", "horses", "racehorse", "stallion","thoroughbred"], 
    "motorbike": ["motorbike", "bicycle", "bikes", "mountain bike","moped"], "player": ["person", "someone", "persons", "woman","somebody"], 
    "potted plant": ["potted plant", "plants", "Plant", "factory","paperboard mill"], "sheep": ["sheep", "lambs", "cows", "goats","cattle"], 
    "sofa": ["sofa", "couch", "settee", "sofas","loveseat"], "train": ["train", "trains", "Train", "commuter train","locomotive"], 
    "tv monitor": ["tv monitor", "monitoring", "monitors", "monitored","laptop computers"]}
elif args.corpus == "coco":
    print("coco")
    prompts = {
    'player': ['player','person','individual','human','people'],
    'bicycle': ['bicycle','bike', 'cycle', 'pedal bike', 'two-wheeler'], 
    'bird': ['bird','avian', 'fowl', 'feathered friend','winged creature'], 
    'boat': ['boat','ship','vessel','watercraft','Canoe'],  
    'truck': ['Pickup', 'Vehicle', 'Van', 'Lorry', 'Semi-truck'],
    'bus': ['bus', 'coach','transit','omnibus', 'motorbus'], 
    'car' : ['car','automobile','vehicle','sedan','carriage'], 
    'cat': ['cat', 'feline', 'kitty', 'tomcat','tabby'],  
    'cow': ['cow','bovine','heifer','bull','ox'], 
    'dog' : ['dog','canine','puppy','hound','pooch'],
    'horse': ['horse','equine','mare','stallion','pony'], 
    'potted plant': ['potted plant','houseplant','flowerpot','planter','indoor plant'], 
    'sheep': ['sheep','lamb','ewe','ram','flock'],  
    'train': ['train','railway','locomotive','subway','monorail'],        
    'motorcycle': ['Bike', 'Scooter', 'Motorbike', 'Moped', 'Two-wheeler'],
    'airplane': ['Aircraft', 'Plane', 'Jet', 'Helicopter', 'Airliner'],
    'traffic light': ['Stoplight', 'Signal', 'Semaphore', 'Traffic signal', 'Red light'],
    'fire hydrant': ['Hydrant', ' Fireplug', ' Water valve', ' Fire valve', ' Fire post'],
    'stop sign': ['Traffic sign', ' Road sign', ' Intersection sign', ' Yield sign', ' Warning sign'],
    'parking meter': ['Meter', ' Parking pay station', ' Coin-operated meter', ' Parking ticket machine', ' Parking payment kiosk'],
    'bench': ['Seat', ' Park bench', ' Pew', ' Stool', ' Settee'],
    'elephant': ['Mammal', ' African elephant', ' Animal', ' Tusked', ' Trunk'],
    'bear': ['Mammal', ' Grizzly bear', ' Animal', ' Brown bear', ' Polar bear'],
    'zebra': ['Mammal', ' Animal', ' Equine', ' Striped horse', ' Wildlife'],
    'giraffe': ['Mammal', ' African giraffe', ' Animal', ' Long-necked', ' Wildlife'],
    'backpack': ['Bag', ' Rucksack', ' Knapsack', ' Satchel', ' Haversack'],
    'umbrella': ['Parasol', ' Canopy', ' Sunshade', ' Brolly', ' Rainshade'],
    'handbag': ['Purse', ' Tote', ' Clutch', ' Satchel', ' Shoulder bag'],
    'tie': ['Necktie', ' Cravat', ' Bow tie', ' Ascot', ' Neckwear'],
    'suitcase': ['Luggage', ' Travel bag', ' Trunk', ' Carry-on', ' Trolley case'],
    'frisbee': ['Flying disc', ' Disc toss', ' Disc golf', ' Ultimate disc', ' Disc sports'],
    'skis': ['Skiing', ' Ski equipment', ' Alpine skiing', ' Snow skiing', ' Ski gear'],
    'snowboard': ['Snowboarding', 'Snowboarder', 'Freestyle snowboarding', 'Snowboard equipment', 'Snowboard gear'],
    'sports ball': ['Ball', 'Soccer ball', 'Basketball', 'Volleyball', 'Tennis ball'],
    'kite': ['Flying kite', 'Kite-flying', 'Kiteboarding', 'Kite flyer', 'Kite runner'],
    'baseball bat': ['Bat', 'Softball bat', 'Wooden bat', 'Aluminum bat', 'Batting club'],
    'baseball glove': ['Glove', 'Mitt', 'Fielding glove', "Catcher's glove", "Infielder's glove"],
    'skateboard': ['Skateboarding', 'Skate', 'Longboard', 'Skatepark', 'Skateboarder'],
    'surfboard': ['Surfing', 'Board', 'Longboard', 'Wave riding', 'Surfer'],
    'tennis racket': ['Racket', 'Tennis racquet', 'Tennis equipment', 'Racquet sport', 'Tennis player'],
    'bottle': ['Container', 'Flask', 'Water bottle', 'Vessel', 'Glass bottle'],
    'wine glass': ['Glass', 'Stemware', 'Goblet', 'Wine goblet', 'Wine vessel'],
    'cup': ['Mug', 'Glass', 'Tumbler', 'Teacup', 'Beverage container'],
    'fork': ['Utensil', 'Silverware', 'Dinner fork', 'Table fork', 'Eating utensil'],
    'knife': ['Blade', 'Cutlery', 'Kitchen knife', "Chef's knife", 'Sharp tool'],
    'spoon': ['Utensil', 'Silverware', 'Tablespoon', 'Dessert spoon', 'Soup spoon'],
    'bowl.': ['Dish', 'Serving bowl', 'Salad bowl', 'Soup bowl', 'Mixing bowl'],
    'banana': ['Fruit', 'Plantain', 'Yellow fruit', 'Tropical fruit', 'Banana peel'],
    'apple': ['Fruit', 'Red apple', 'Green apple', 'Apple tree', 'Apple orchard'],
    'sandwich': ['Sub', 'Wrap', 'Hoagie', 'Burger', 'Deli sandwich'],
    'orange': ['Fruit', 'Citrus', 'Tangerine', 'Mandarin', 'Orange peel'],
    'broccoli': ['Vegetable', 'Cauliflower', 'Brussels sprouts', 'Green vegetable', 'Leafy green'],
    'carrot': ['Vegetable', 'Root vegetable', 'Orange vegetable', 'Carrot stick', 'Carrot juice'],
    'hot dog': ['Frankfurter', 'Sausage', 'Wiener', 'Grilled hot dog', 'Hot dog bun'],
    'pizza': ['Slice', 'Cheese pizza', 'Pepperoni pizza', 'Margherita pizza', 'Pizza delivery'],
    'donut': ['Pastry', 'Doughnut', 'Glazed donut', 'Sprinkled donut', 'Jelly-filled donut'],
    'cake': ['Dessert', 'Birthday cake', 'Chocolate cake', 'Layer cake', 'Wedding cake'],
    'chair': ['Seat', 'Furniture', 'Armchair', 'Recliner', 'Stool'],
    'couch': ['Sofa', 'Loveseat', 'Sectional', 'Settee', 'Chesterfield'],
    'potted plant': ['Indoor plant', 'Houseplant', 'Container plant', 'Flowerpot', 'Greenery'],
    'bed': ['Mattress', 'Bedroom', 'Sleep', 'Bedding', 'Bunk bed'],
    'dining table': ['Dining room table', 'Kitchen table', 'Wooden table', 'Dinner table', 'Round table'],
    'toilet': ['Bathroom', 'Restroom', 'Commode', 'Lavatory', 'WC (Water Closet)'],
    'tv': ['Television', 'Television set', 'Flat screen', 'Smart TV', 'HDTV'],
    'laptop': ['Notebook', 'Computer', 'Portable computer', 'Laptop computer', 'Personal computer'],
    'mouse': ['Computer mouse', 'Wireless mouse', 'Optical mouse', 'Gaming mouse', 'Trackpad'],
    'remote': ['Remote control', 'TV remote', 'Wireless remote', 'Universal remote', 'Infrared remote'],
    'keyboard': ['Computer keyboard', 'Mechanical keyboard', 'Wireless keyboard', 'Gaming keyboard', 'QWERTY keyboard'],
    'cell phone': ['Mobile phone', 'Smartphone', 'Cellular phone', 'iPhone', 'Android phone'],
    'microwave': ['Microwave oven', 'Countertop microwave', 'Appliance', 'Kitchen appliance', 'Cooking device'],
    'oven': ['Kitchen oven', 'Electric oven', 'Stove', 'Range oven', 'Baking oven'],
    'toaster': ['Toast machine', 'Bread toaster', 'Toaster oven', 'Pop-up toaster', 'Bread toaster'],
    'sink': ['Kitchen sink', 'Bathroom sink', 'Washbasin', 'Basin', 'Vanity sink'],
    'refrigerator': ['Fridge', 'Refrigeration', 'Freezer', 'Refrigeration unit', 'Cooling appliance'],
    'book': ['Novel', 'Paperback', 'Literature', 'Hardcover', 'Textbook'],
    'clock': ['Wall clock', 'Alarm clock', 'Timepiece', 'Grandfather clock', 'Digital clock'],
    'vase': ['Flower vase', 'Decorative vase', 'Ceramic vase', 'Glass vase', 'Ornamental vase'],
    'scissors': ['Shears', 'Cutting tool', 'Snips', 'Clippers', 'Trimming scissors'],
    'teddy bear': ['Stuffed animal', 'Plush bear', 'Toy bear', 'Cuddly bear', 'Teddy toy'],
    'hair drier': ['Blow dryer', 'Hairdryer', 'Hair styling tool', 'Hair blower', 'Salon dryer'],
    'toothbrush': ['Dental brush', 'Tooth cleaner', 'Oral hygiene tool', 'Tooth scrubber', 'Dental care brush']
}
elif args.corpus == 'GPT_v1_4':
    print("GPT_v1_4")
    prompts = {'aeroplane':['aeroplane', 'Aircraft', 'Airplane', 'Plane'], 
            'bicycle': ['bicycle','Bike', 'Cycle', 'Pedal bike'], 
            'bird': ['bird','Avian', 'Fowl', 'Feathered friend'], 
            'boat': ['boat','Ship','Vessel','Watercraft'], 
            'bottle': ['bottle', 'Flask','Container','Jar'], 
            'bus': ['bus', 'Coach','Transit','Omnibus'], 
            'car' : ['car','Automobile','Vehicle','Sedan'], 
            'cat': ['cat', 'Feline', 'Kitty', 'Tomcat'], 
            'chair': ['chair', 'Seat','Armchair','Recliner'], 
            'cow': ['cow','Bovine','Heifer','Bull'], 
            'dining table': ['dining table','Kitchen table','Dinner table','Breakfast table'],
            'dog' : ['dog','Canine','Puppy','Hound'],
            'horse': ['horse','Equine','Mare','Stallion'], 
            'motorbike': ['motorbike','Motorcycle','Bike','Scooter'], 
            'player': ['person','individual','Human','People'], 
            'potted plant': ['potted plant','Houseplant','Flowerpot','Planter'], 
            'sheep': ['sheep','Lamb','Ewe','Ram'], 
            'sofa': ['sofa','Couch','Loveseat','Settee'], 
            'train': ['train','Railway','Locomotive','Subway'], 
            'tv monitor': ['tv monitor','Television','Display screen','Flat screen']}
elif args.corpus == 'GPT_v1_2ch':
    print("GPT_v1_2ch")
    prompts = {'aeroplane':['aeroplane', 'aircraft'], 'bicycle': ['bicycle','bike'], 'bird': ['bird','avian'], 'boat': ['boat','ship'], 
            'bottle': ['bottle', 'flask'], 'bus': ['bus', 'coach'], 'car' : ['car','automobile'], 'cat': ['cat', 'feline'], 'chair': ['chair', 'seat'], 
            'cow': ['cow','bovine'], 'dining table': ['dining table','kitchen table'], 'dog' : ['dog','canine'], 'horse': ['horse','equine'], 
            'motorbike': ['motorbike','motorcycle'], 'player': ['player','person'], 'potted plant': ['potted plant','houseplant'], 
            'sheep': ['sheep','lamb'], 'sofa': ['sofa','couch'], 'train': ['train','railway'], 'tv monitor': ['tv monitor','television']}
elif args.corpus == 'GPT_v1_3ch':
    print("GPT_v1_3ch")
    prompts = {'aeroplane':['aeroplane', 'aircraft', 'airplane'], 'bicycle': ['bicycle','bike', 'cycle'], 'bird': ['bird','avian', 'fowl'], 
            'boat': ['boat','ship','vessel'], 'bottle': ['bottle', 'flask','container'], 'bus': ['bus', 'coach','transit'], 
            'car' : ['car','automobile','vehicle'], 'cat': ['cat', 'feline', 'kitty'], 'chair': ['chair', 'seat','armchair'], 
            'cow': ['cow','bovine','heifer'], 'dining table': ['dining table','kitchen table','dinner table'], 'dog' : ['dog','canine','puppy'],
            'horse': ['horse','equine','mare'], 'motorbike': ['motorbike','motorcycle','bike'], 'player': ['player','person','individual'], 
            'potted plant': ['potted plant','houseplant','flowerpot'], 'sheep': ['sheep','lamb','ewe'], 'sofa': ['sofa','couch','loveseat'], 
            'train': ['train','railway','locomotive'], 'tv monitor': ['tv monitor','television','display screen']}

elif args.corpus == 'GPT_v1_4ch':
    print("GPT_v1_4ch")
    prompts = {'aeroplane':['aeroplane', 'aircraft', 'airplane', 'plane'], 
            'bicycle': ['bicycle','bike', 'cycle', 'pedal bike'], 
            'bird': ['bird','avian', 'fowl', 'feathered friend'], 
            'boat': ['boat','ship','vessel','watercraft'], 
            'bottle': ['bottle', 'flask','container','jar'], 
            'bus': ['bus', 'coach','transit','omnibus'], 
            'car' : ['car','automobile','vehicle','sedan'], 
            'cat': ['cat', 'feline', 'kitty', 'tomcat'], 
            'chair': ['chair', 'seat','armchair','recliner'], 
            'cow': ['cow','bovine','heifer','bull'], 
            'dining table': ['dining table','kitchen table','dinner table','breakfast table'],
            'dog' : ['dog','canine','puppy','hound'],
            'horse': ['horse','equine','mare','stallion'], 
            'motorbike': ['motorbike','motorcycle','bike','scooter'], 
            'player': ['player','person','individual','human'], 
            'potted plant': ['potted plant','houseplant','flowerpot','planter'], 
            'sheep': ['sheep','lamb','ewe','ram'], 
            'sofa': ['sofa','couch','loveseat','settee'], 
            'train': ['train','railway','locomotive','subway'], 
            'tv monitor': ['tv monitor','television','display screen','flat screen']}

elif args.corpus == 'GPT_v1_5ch':
    print("GPT_v1_5ch")
    prompts = {'aeroplane':['aeroplane', 'aircraft', 'airplane', 'plane', 'jet'], 
            'bicycle': ['bicycle','bike', 'cycle', 'pedal bike', 'two-wheeler'], 
            'bird': ['bird','avian', 'fowl', 'feathered friend','winged creature'], 
            'boat': ['boat','ship','vessel','watercraft','Canoe'], 
            'bottle': ['bottle', 'flask','container','jar','vial'], 
            'bus': ['bus', 'coach','transit','omnibus', 'motorbus'], 
            'car' : ['car','automobile','vehicle','sedan','carriage'], 
            'cat': ['cat', 'feline', 'kitty', 'tomcat','tabby'], 
            'chair': ['chair', 'seat','armchair','recliner','stool'], 
            'cow': ['cow','bovine','heifer','bull','ox'], 
            'dining table': ['dining table','kitchen table','dinner table','breakfast table','banquet table'],
            'dog' : ['dog','canine','puppy','hound','pooch'],
            'horse': ['horse','equine','mare','stallion','pony'], 
            'motorbike': ['motorbike','motorcycle','bike','scooter','motor scooter'], 
            'player': ['player','person','individual','human','people'], 
            'potted plant': ['potted plant','houseplant','flowerpot','planter','indoor plant'], 
            'sheep': ['sheep','lamb','ewe','ram','flock'], 
            'sofa': ['sofa','couch','loveseat','settee','sectional'], 
            'train': ['train','railway','locomotive','subway','monorail'], 
            'tv monitor': ['tv monitor','television','display screen','flat screen','computer monitor']}

elif args.corpus == 'Across':
    print("Across")
    prompts = {'aeroplane':['aeroplane', 'aircraft','airplane','biplane'], 'bicycle': ['bicycle','bicycles','handlebar','tricycle'], 
            'bird': ['bird','finch', 'skylark', 'owl'], 'boat': ['boat','ship','yacht','vessel'], 
            'bottle': ['bottle','jug','jar','bottles'], 'bus': ['bus','minibus','taxi','buses'], 
            'car' : ['car','vehicle','automobile','van'], 'cat': ['cat', 'kitten','tabby','cats'], 
            'chair': ['chair','chairs','upholstered','sit'], 'cow': ['cow','calve','sheep','milking'], 
            'dining table': ['dining table','table','glass-topped','marble-topped'],'dog' : ['dog','puppy','pitbull','alsatian'],
            'horse': ['horse','stallion','racehorse','mare'], 'motorbike': ['motorbike','motorcycle','scooter','bike'], 
            'player': ['player','person','people','woman'], 'potted plant': ['potted plant','plant','herbaceous','flower'], 
            'sheep': ['sheep','lambs','goats','goat'], 'sofa': ['sofa','couch','settee', 'armchair'], 
            'train': ['train','tram','freight','Intercity'], 'tv monitor': ['tv monitor','monitor','monitors','svga']}
elif args.corpus == 'Google':
    print("Google News")
    prompts = {"aeroplane": ["aeroplane", "Tunng", "Martin Solveig", "Alessi Ark"], 
    "bicycle": ["bicycle", "bike", "scooter", "bicycles"], 
    "bird": ["bird", "birds", "raptor", "owl"], "boat": ["boat", "boats", "sailboat", "motorboat"], 
    "bottle": ["bottle", "bottles", "jug", "corked bottle"], "bus": ["bus", "buses", "Bus", "busses"], 
    "car": ["car", "vehicle", "cars", "SUV"], "cat": ["cat", "cats", "dog", "kitten"], 
    "chair": ["chair", "chairs", "Chair", "chairperson"], 
    "cow": ["cow", "cows", "pig", "dairy cow"], 
    "dining table": ["dining table", "tables", "tray", "dining room"], 
    "dog": ["dog", "dogs", "puppy", "pitbull"], "horse": ["horse", "horses", "racehorse", "stallion"], 
    "motorbike": ["motorbike", "bicycle", "bikes", "mountain bike"], "player": ["person", "someone", "persons", "woman"], 
    "potted plant": ["potted plant", "plants", "Plant", "factory"], "sheep": ["sheep", "lambs", "cows", "goats"], 
    "sofa": ["sofa", "couch", "settee", "sofas"], "train": ["train", "trains", "Train", "commuter train"], 
    "tv monitor": ["tv monitor", "monitoring", "monitors", "monitored"]}
elif args.corpus == 'Wiki_2ch':
    print("English Wikipedia")
    prompts = {"aeroplane": ["aeroplane", "airplane"], 
    "bicycle": ["bicycle", "bike"], 
    "bird": ["bird", "birds"], 
    "boat": ["boat", "motorboat"], 
    "bottle": ["bottle", "flask"],
     "bus": ["bus", "buse"], 
     "car": ["car", "vehicle"], 
     "cat": ["cat", "dog"], 
     "chair": ["chair", "chairman"], 
     "cow": ["cow", "goat"], 
     "dining table": ["dining table", "tables"], 
     "dog": ["dog", "puppy"], 
     "horse": ["horse", "hors"], 
     "motorbike": ["motorbike", "bike"], 
     "player": ["player", "persons"], 
     "potted plant": ["potted plant", "plant"], 
     "sheep": ["sheep", "cattle"], 
     "sofa": ["sofa", "couch"], 
     "train": ["train", "intercity"], 
     "tv monitor": ["tv monitor", "monitor"]}
elif args.corpus == 'Wiki_3ch':
    print("English Wikipedia")
    prompts = {"aeroplane": ["aeroplane", "airplane", "aircraft"], 
    "bicycle": ["bicycle", "bike","motorcycle"], 
    "bird": ["bird", "birds", "reptile"], 
    "boat": ["boat", "motorboat", "yacht"], 
    "bottle": ["bottle", "flask", "jug"],
     "bus": ["bus", "buse", "tram"], 
     "car": ["car", "vehicle", "truck"], 
     "cat": ["cat", "dog", "kitten"], 
     "chair": ["chair", "chairman", "chairperson"], 
     "cow": ["cow", "goat", "pig"], 
     "dining table": ["dining table", "tables", "table"], 
     "dog": ["dog", "puppy", "cat"], 
     "horse": ["horse", "hors", "racehorse"], 
     "motorbike": ["motorbike", "bike", "bicycle"], 
     "player": ["player", "persons", "individual"], 
     "potted plant": ["potted plant", "plant", "flower"], 
     "sheep": ["sheep", "cattle", "goat"], 
     "sofa": ["sofa", "couch", "cupboard"], 
     "train": ["train", "intercity", "freight"], 
     "tv monitor": ["tv monitor", "monitor", "monitoring"]}

elif args.corpus == 'Wiki_4ch':
    print("English Wikipedia")
    prompts = {"aeroplane": ["aeroplane", "airplane", "aircraft", "biplane"], 
    "bicycle": ["bicycle", "bike", "motorcycle", "motorbike"], 
    "bird": ["bird", "birds", "reptile", "waterfowl"], 
    "boat": ["boat", "motorboat", "yacht", "barge"], 
    "bottle": ["bottle", "flask", "jug", "soda"],
     "bus": ["bus", "buse", "tram", "trolleybus"], 
     "car": ["car", "vehicle", "truck", "automobile"], 
     "cat": ["cat", "dog", "kitten", "rabbit"], 
     "chair": ["chair", "chairman", "chairperson", "chairwoman"], 
     "cow": ["cow", "goat", "pig", "sheep"], 
     "dining table": ["dining table", "tables", "table", "below"], 
     "dog": ["dog", "puppy", "cat", "pet"], 
     "horse": ["horse", "hors", "racehorse", "thoroughbred"], 
     "motorbike": ["motorbike", "bike", "bicycle", "motorbike"], 
     "player": ["player", "persons", "individual", "person"], 
     "potted plant": ["potted plant", "plant", "flower", "herbaceous"], 
     "sheep": ["sheep", "cattle", "goat", "pig"], 
     "sofa": ["sofa", "couch", "cupboard", "bathtub"], 
     "train": ["train", "intercity", "freight", "tram"], 
     "tv monitor": ["tv monitor", "monitor", "monitoring", "monitore"]}
elif args.corpus == 'Wiki_5ch':
    print("English Wikipedia")
    prompts = {"aeroplane": ["aeroplane", "airplane", "aircraft", "biplane", "airliner"], 
    "bicycle": ["bicycle", "bike", "motorcycle", "motorbike", "scooter"], 
    "bird": ["bird", "birds", "reptile", "waterfowl", "insect"], 
    "boat": ["boat", "motorboat", "yacht", "barge"], 
    "bottle": ["bottle", "flask", "jug", "soda", "carton"],
     "bus": ["bus", "buse", "tram", "trolleybus", "trolleybuse"], 
     "car": ["car", "vehicle", "truck", "automobile", "suv"], 
     "cat": ["cat", "dog", "kitten", "rabbit", "raccoon"], 
     "chair": ["chair", "chairman", "chairperson", "chairwoman"], 
     "cow": ["cow", "goat", "pig", "sheep", "cattle"], 
     "dining table": ["dining table", "tables", "table", "below", "sofa"], 
     "dog": ["dog", "puppy", "cat", "pet", "pig"], 
     "horse": ["horse", "hors", "racehorse", "thoroughbred", "donkey"], 
     "motorbike": ["motorbike", "bike", "bicycle", "motorbike", "motorcycle"], 
     "player": ["player", "persons", "individual", "person"], 
     "potted plant": ["potted plant", "plant", "flower", "herbaceous", "cacti"], 
     "sheep": ["sheep", "cattle", "goat", "pig", "cow"], 
     "sofa": ["sofa", "couch", "cupboard", "bathtub", "armchair"], 
     "train": ["train", "intercity", "freight", "tram", "trainset"], 
     "tv monitor": ["tv monitor", "monitor", "monitoring", "monitore", "detect"]}
else:
    print("English Gigaword")
    prompts = {"aeroplane": ["aeroplane", "airplane", "aircraft", "plane"], "bicycle": ["bicycle", "bike", "motorbike", "motorcycle"], "bird": ["bird", "fowl", "avian", "h5n1"], "boat": ["boat", "vessel", "ship", "sailboat"], "bottle": ["bottle", "carton", "jug", "jar"], "bus": ["bus", "minibus", "train", "taxi"], "car": ["car", "vehicle", "suv", "minivan"], "cat": ["cat", "dog", "raccoon", "pet"], "chair": ["chair", "sofa", "Chair", "sit"], "cow": ["cow", "sheep", "cattle", "pig"], "dining table": ["dining table", "top", "sit", "folding"], "dog": ["dog", "cat", "pet", "puppy"], "horse": ["horse", "thoroughbred", "racehorse", "stallion"], "motorbike": ["motorbike", "bicycle", "scooter", "biking"], "player": ["player","someone", "individual", "people"], "potted plant": ["potted plant", "factory", "refinery", "coal-fired"], "sheep": ["sheep", "goat", "pig", "cattle"], "sofa": ["sofa", "couch", "armchair", "recline"], "train": ["train", "bus", "rail", "tram"], "tv monitor": ["tv monitor", "monitoring", "monitor", "observer"]}

#import json
#with open("/home/ruxie/scratch/ruxie/GPT_bbox_margin_prompts.json", "rb") as f:
    #labels = json.load(f)

with open('/home/ruxie/projects/def-josedolz/ruxie/CLIP-WSL/voc12/train_aug.txt') as file:
    lines = [line.rstrip() for line in file]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

path_img = "/home/ruxie/projects/def-josedolz/ruxie/Data/VOCdevkit/VOC2012/JPEGImages"
path_cam = "/home/ruxie/scratch/ruxie/baseline/cam_mask"

master_dict = {}
for name in tqdm(lines,total=len(lines)):
    image_name = name
    #print(image_name)
    #label_image = labels[image_name]['classes']

    image = cv2.imread(os.path.join(path_img, image_name +'.jpg'))
    cam = np.load(os.path.join(path_cam,image_name +'.npy'), allow_pickle = True).item()

    label_image = cam['keys']
    class_labels = []
    for j in label_image:
        class_labels.append(voc[j])
    
    cam = cam['high_res']

    label_probs =[]
    for i in range(cam.shape[0]):
        #temp = prompts[class_labels[i]]
        temp = [f"A photo of {c}." for c in prompts[class_labels[i]]]
        print(temp)
        #print(temp)
        cam_new = cam[i]
        cam_img = image.transpose(2, 0, 1) * cam_new
        cam_img = cam_img.transpose(1,2,0)*255.0
        cam_img = np.array(cam_img, dtype=np.uint8)

        image_a = preprocess(NumpyToPil(cam_img)).unsqueeze(0).to(device)
        text = clip.tokenize(temp).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_a)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image_a, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_list = probs[0]    
        label_probs.append(temp[np.argmax(probs_list)])
        master_dict[image_name.replace("_", "")] = label_probs

with open('/home/ruxie/scratch/ruxie/wiki_4ch.json', 'wb') as f:
    pkl.dump(master_dict, f)
