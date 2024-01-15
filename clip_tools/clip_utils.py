import torch
import pickle
from net.resnet50 import Adapter, Textadapter

category_dict = {
    'voc': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
            'dog',
            'horse', 'motorbike', 'player', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'],
    'coco': ['player', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']
}

background_dict = {
    'voc': ['a photo of tree.', 'a photo of river.',
            'a photo of sea.', 'a photo of lake.', 'a photo of water.',
            'a photo of railway.', 'a photo of railroad.', 'a photo of track.',
            'a photo of stone.', 'a photo of rocks.'],
    'coco': ['a photo of tree.', 'a photo of river.',
             'a photo of sea.', 'a photo of lake.', 'a photo of water.',
             'a photo of railway.', 'a photo of railroad.', 'a photo of track.',
             'a photo of stone.', 'a photo of rocks.', 'a photo of playground.', 'a photo of spray.'],
}
foreground_dict = {'aeroplane': ['a photo of aircraft.','a photo of airplane.','a photo of plane.'], 'bicycle' :['a photo of cycle.', 'a photo of bike.','a photo of motorbike.'],'bird':['a photo of canary.','a photo of reptile.','a photo of parrot.'], 'boat':['a photo of ship.','a photo of yatch.', 'a photo of vessel.'], 'bottle':['a photo of flask.','a photo of jug.','a photo of glass.'], 'bus':['a photo of vehicle.','a photo of taxi.','a photo of van.'], 'car':['a photo of truck.', 'a photo of automobile.',' a photo of jeep.'], 'cat':['a photo of dog.', 'a photo of kitten.','a photo of pet.'], 'chair':['a photo of sofa.','a photo of table.', 'a photo of arm chair.'], 'cow':['a photo of goat.','a phtot of pig.', 'a photo of buffalo.'], 'dining table':['a photo of table.','a photo of billiard.','a photo of desk.'],
            'dog':['a photo of puppy.', 'a photo of cat.', 'a photo of animal.'],'horse':['a photo of donkey.', 'a photo of oxen.', 'a photo of racehorse.'], 'motorbike':['a photo of motorcycle.', 'a photo of bike.','a photo of bicycle.'], 'player':['a photo of goalkeeper.', 'a photo of batsman.','a photo of person.'], 'potted plant':['a photo of shrub.','a photo of cacti.', 'a photo of plant.'], 'sheep':['a photo of cattle.', 'a photo of goat.','a photo of lamb.'], 'sofa':['a photo of couch.', 'a photo of cupboard.',' a photo of settee.'], 'train':['a photo of locomotive.', ' a photo of tram.',' a photo of rail.'], 'tv monitor':['a photo of computer.','a photo of television.', 'a photo of computer monitor.'] }

best_sen  = {'aeroplane': ['a photo of airplane.'], 'bicycle' :['a photo of bicycle.'],'bird':['a photo of canary.'], 'boat':['a photo of ship.'], 'bottle':['a photo of bottle.'], 'bus':['a photo of vehicle.'], 'car':['a photo of car.'], 'cat':['a photo of cat'], 'chair':['a photo of sofa.'], 'cow':['a photo of cow.'], 'dining table':['a photo of table.'],
            'dog':['a photo of dog.'],'horse':['a photo of horse.'], 'motorbike':['a photo of motorbike.'], 'player':['a photo of player.'], 'potted plant':['a photo of potted plant.'], 'sheep':['a photo of sheep.'], 'sofa':['a photo of sofa.'], 'train':[' a photo of tram.'], 'tv monitor':['a photo of tv monitor.'] }

ablation_dict = {'aeroplane': ['a photo of biplane.','a photo of aeroplane.','a photo of airplane.','a photo of airplane.'], 'bicycle' :['a photo of bike.','a photo of bike.','a photo of bicycle.','a photo of bike.'],'bird':['a photo of finch.','a photo of birds.','a photo of birds.','a photo of avian.'], 'boat':['a photo of yatch.','a photo of boats.','a photo of motorboat.','a photo of vessel.'], 'bottle':['a photo of jar.','a photo of bottles.','a photo of bottle.','a photo of jug.'], 'bus':['a photo of bus.','a photo of buses.','a photo of bus.','a photo of minibus.'], 'car':['a photo of vehicle.','a photo of vehicle.','a photo of vehicle.','a photo of vehicle.'], 'cat':['a photo of kitten.','a photo of cats.','a photo of cat.','a photo of pet.'], 'chair':['a photo of sofa.','a photo of chairs.','a photo of chair.','a photo of sofa.'], 'cow':['a photo of cattle.','a photo of cows.','a photo of cow.','a photo of cattle.'], 'dining table':['a photo of sideboard.','a photo of tables.','a photo of tables.','a photo of desk.'],
            'dog':['a photo of puppy.','a photo of dogs.','a photo of puppy.','a photo of puppy.'],'horse':['a photo of stallion.','a photo of horses.','a photo of horse.','a photo of racehorse.'], 'motorbike':['a photo of motorcycle.','a photo of motorbikes.','a photo of motorcycle.','a photo of motorcycle.'], 'player':['a photo of player.','a photo of persons.','a photo of persons.','a photo of people.'], 'potted plant':['a photo of potted plant.','a photo of plants.','a photo of plant.','a photo of plant.'], 'sheep':['a photo of goat.','a photo of lambs.','a photo of goat.','a photo of goat.'], 'sofa':['a photo of armchair.','a photo of couch.','a photo of couch.','a photo of couch.'], 'train':['a photo of railway.','a photo of trains.','a photo of freight.','a photo of tram.'], 'tv monitor':['a photo of tv monitor.','a photo of tv monitor.','a photo of tv monitor.','a photo of tv monitor.'] }

prompt_dict = ['a photo of {}.']


def to_text(labels, dataset='voc'):
    _d = category_dict[dataset]

    text = []
    for i in range(labels.size(0)):
        idx = torch.nonzero(labels[i], as_tuple=False).squeeze()
        if torch.sum(labels[i]) == 1:
            idx = idx.unsqueeze(0)
        cnt = idx.shape[0] - 1
        if cnt == -1:
            text.append('background')
        elif cnt == 0:
            text.append(prompt_dict[cnt].format(_d[idx[0]]))
        elif cnt == 1:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]]))
        elif cnt == 2:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]]))
        elif cnt == 3:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]]))
        elif cnt == 4:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]], _d[idx[4]]))
        else:
            raise NotImplementedError
    return text

def get_texts_for_image(texts,sen):

        upd_texts = []

        for idx in texts:
            #print(idx)
            dict_key = idx.replace('a photo of ','').replace('.','')
            #print(key)
            if sen=='best':
                temp = best_sen[dict_key]
                upd_texts.append(temp[0])
                #print("hii")
            else:
                #print("ohno")
                temp = foreground_dict[dict_key]
                if sen == 'key':
                    upd_texts.append(idx)
                elif sen == 'sen1':
                    upd_texts.append(temp[0])
                elif sen == 'sen2':
                    upd_texts.append(temp[1])
                else: 
                    upd_texts.append(temp[2])

         #print(upd_texts)

        return upd_texts

def get_texts_for_image_ab(texts,sen):

        upd_texts = []

        for idx in texts:
            #print(idx)
            dict_key = idx.replace('a photo of ','').replace('.','')
            #print(key)
            temp = ablation_dict[dict_key]
            if sen == 'BC':
                upd_texts.append(temp[0])
            elif sen == 'AN':
                upd_texts.append(temp[1])
            elif sen == 'AW':
                upd_texts.append(temp[2])
            else: 
                upd_texts.append(temp[3])

         #print(upd_texts)

        return upd_texts

def get_best_prompts_from_json(file):
    if file == 'British':
        #print("British National Corpus")
        with open('/home/ruxie/scratch/ruxie/British_national_corpus.json','rb') as f:
            data = pickle.load(f)
    elif file == 'Across':
        #print("Google News")
        with open('/home/ruxie/scratch/ruxie/Across.json','rb') as f:
            data = pickle.load(f)
    elif file == 'GPT_4':
        print("GPT_4_2ch")
        with open('/home/ruxie/scratch/ruxie/GPT_v1_A.json','rb') as f:
            data = pickle.load(f)
    elif file == 'GPT_4_1':
        print("GPT_4_1_3ch")
        with open('/home/ruxie/scratch/ruxie/GPT_v1_A_scene.json','rb') as f:
            data = pickle.load(f)
    elif file == 'GPT_4_2':
        print("wiki_4ch")
        with open('/home/ruxie/scratch/ruxie/GPT_v1_ch5.json','rb') as f:
            data = pickle.load(f)
    elif file == 'GPT_4_3':
        print("GPT_4_3")
        with open('/home/ruxie/scratch/ruxie/google_ch5.json','rb') as f:
            data = pickle.load(f)
    elif file == 'Google':
        #print("Google News")
        with open('/home/ruxie/scratch/ruxie/Google_news.json','rb') as f:
            data = pickle.load(f)
    elif file == 'Wiki':
        #print("English Wikipedia")
        with open('/home/ruxie/scratch/ruxie/English_wikipedia.json','rb') as f:
            data = pickle.load(f)
    else:
        #print("English Gigaword")
        with open('/home/ruxie/scratch/ruxie/English_Gigaword.json','rb') as f:
            data = pickle.load(f)
    return data



import clip
def clip_forward(clip_model, images, labels, dname='coco',sen='key',fnames=None, file = 'British', adapter=None):
    #print(labels)
    texts = to_text(labels, dname)
    #print(texts)
    #texts = clip.tokenize(texts).cuda()
    #print(file)
    original = False
    #mode = False
    
    if not original:
        mode = False
        if mode:
            json_map = get_best_prompts_from_json(file)

            upd_texts = []
        
            for fname in fnames:
                #upd_texts += ['a photo of {}.'.format(ii) for ii in json_map[fname.replace('_','')]]
                upd_texts += ['{}'.format(ii) for ii in json_map[fname.replace('_','')]]
        
        #print (upd_texts)

        if not mode: foreground_texts = get_texts_for_image(texts,sen)

        print("fore ",foreground_texts) 
        if not mode:foreground_texts = clip.tokenize(foreground_texts).cuda()
        else: 
            print('hey i am here')
            foreground_texts = clip.tokenize(upd_texts).cuda() 
    else: foreground_texts = clip.tokenize(texts).cuda()
    '''
    mode = False
    #mode = True
    
    if mode:
        json_map = get_best_prompts_from_json()

        upd_texts = []
    
        for fname in fnames:
            upd_texts += ['a photo of {}.'.format(ii) for ii in json_map[fname.replace('_','')]]
            #upd_texts += ['{}'.format(ii) for ii in json_map[fname.replace('_','')]]
    
    #print ("fore",upd_texts)

    if not mode: foreground_texts = get_texts_for_image_ab(texts,sen)

    #print("fore ",foreground_texts)

    #,"upd ",upd_texts)
    if not mode:foreground_texts = clip.tokenize(foreground_texts).cuda()
    else: foreground_texts = clip.tokenize(upd_texts).cuda()
    '''
    image_features = clip_model.encode_image(images)
    #text_features = clip_model.encode_text(texts)
    text_features = clip_model.encode_text(foreground_texts)

    
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    if adapter is not None:
        text_features = adapter(text_features)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)
    text_features = text_features.reshape(N, C, 1)

    similarity = torch.matmul(image_features, text_features)
    #print(image_features.shape, text_features.shape, similarity.shape)
    return similarity
