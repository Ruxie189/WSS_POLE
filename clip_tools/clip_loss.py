import torch
import clip
from clip_tools.clip_utils import background_dict, category_dict, to_text

# maximize similarity
class SimMaxLoss(torch.nn.Module):

    def __init__(self, margin=0.0):
        super(SimMaxLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x, weights):
        #print(self.margin)
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(x + self.margin) * weights).mean()
    '''
    def forward(self, x, weights):
        x += self.margin
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(x) * weights).mean()
    '''
# minimize similarity
class SimMinLoss(torch.nn.Module):

    def __init__(self, margin=0.0):
        super(SimMinLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(1 - x + self.margin) * weights).mean()

def get_background_prompts_from_json():
    with open('/home/ruxie/scratch/ruxie/gpt_margin_sim2_back.json','rb') as f:
        data = pickle.load(f)
    return data

# suppress background activation
class BackgroundSuppressionLoss(torch.nn.Module):
    """
    based on threshold
    """

    def __init__(self, threshold=0.26, dname='coco'):
        super(BackgroundSuppressionLoss, self).__init__()
        self.dname = dname
        self.background = background_dict[dname]
        self.threshold = threshold
        print(f'Use CBSLoss! threshold: {threshold}')

    def forward(self, clip_model, images, dname='coco', fnames=None, eps=0.0001): #pass labels for pascal
        #texts = to_text(labels, dname)
        #json_map = get_background_prompts_from_json()
        #upd_texts = []
    
        #for fname in fnames:
            #upd_texts += ['a photo of {}.'.format(ii) for ii in json_map[fname.replace('_','')]]
            #upd_texts += ['{}'.format(ii) for ii in json_map[fname.replace('_','')]]
         
        #print ("back",upd_texts)

        image_features = clip_model.encode_image(images)  # [N1, C]

        text_features = clip_model.encode_text(clip.tokenize(self.background).cuda())  # [N2, C]
        #text_features = clip_model.encode_text(clip.tokenize(upd_texts).cuda())
        
        # normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = (image_features @ text_features.t())  # [N1, N2]
        #print(logits_per_image)
        mask = torch.zeros_like(logits_per_image)
        mask = torch.where(logits_per_image > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))
        #print(mask)
        #print(-(torch.log(1 - logits_per_image) * mask).sum())
        return -(torch.log(1 - logits_per_image) * mask).sum()



#class ConsistencyLoss(torch.nn.Module):
   #def __init__(self,dname='coco'):
       #super(ConsistencyLoss, self).__init__()
   
   #def forward(self, cam_otm1, cam_otm2):
   #write CE loss after global average pooling 

#foregroundloss
class ForegroundImportanceLoss(torch.nn.Module):
    """
    based on threshold
    """

    def __init__(self,threshold=0.26, dname='coco'):
        super(ForegroundImportanceLoss, self).__init__()

        self.foreground = {'aeroplane': ['a photo of aircraft.','a photo of airplane.','a photo of helicopter.'], 'bicycle' :['a photo of cycle.', 'a photo of bike.','a photo of two wheeler.'],'bird':['a photo of canary.','a photo of reptile.','a photo of sparrow.'], 'boat':['a photo of ship.','a photo of yatch.', 'a photo of motorboat.'], 'bottle':['a photo of flask.','a photo of jug.','a photo of container.'], 'bus':['a photo of vehicle.','a photo of taxi.','a photo of four wheeler.'], 'car':['a photo of truck.', 'a photo of automobile.',' a photo of jeep.'], 'cat':['a photo of dog.', 'a photo of kitten.','a photo of kitty.'], 'chair':['a photo of sofa.','a photo of table.', 'a photo of bench.'], 'cow':['a photo of goat.','a phtot of pig.', 'a photo of cattle.'], 'dining table':['a photo of table.','a photo of billiard.','a photo of dinner table.'],
            'dog':['a photo of puppy.', 'a photo of cat.', 'a photo of pet.'],'horse':['a photo of donkey.', 'a photo of oxen.', 'a photo of foal.'], 'motorbike':['a photo of motorcycle.', 'a photo of bike.','a photo of scooter.'], 'player':['a photo of goalkeeper.', 'a photo of batsman.','a photo of person.'], 'potted plant':['a photo of shrub.','a photo of cacti.', 'a photo of leaves.'], 'sheep':['a photo of cattle.', 'a photo of goat.','a photo of lamb.'], 'sofa':['a photo of couch.', 'a photo of cupboard.',' a photo of lounge.'], 'train':['a photo of locomotive.', ' a photo of tram.',' a photo of rail.'], 'tv monitor':['a photo of computer.','a photo of television.', 'a photo of screen.'] }
        self.foreground_dist = {'aeroplane': [0.815,0.713], 'bicycle': [0.5,0.787], 'bird': [0.4,0.679], 'boat':[0.664, 0.710, 0.718], 'bottle':[0.667,0.632], 'bus':[0.5, 0.611], 'car':[0.661,0.660], 'cat':[0.689,0.613], 'chair':[0.5,0.5], 'cow':[0.743,0.723,0.690], 'dining table':[0.686,'0.398'],'dog': [0.691,0.689,0.678],'horse': [0.636,0.631], 'motorbike' : [0.769,0.744], 'player': [0.632,0.5], 'potted plant': [0.559,0.561], 'sheep':[0.793,0.782], 'sofa':[0.648,0.586],'train':[0.5,0.635],'tv monitor':[0.5,0.810]}  
        self.dname = dname
        self.threshold = threshold
        
    def get_texts_for_image(self,texts):

        upd_texts1 = []
        upd_texts2 = []

        for idx in texts:
            #print(idx)
            key = idx.replace('a photo of ','').replace('.','')
            #print(key)
            temp = self.foreground[key]
            #print("temp",temp)
            upd_texts1.append(temp[2])
            #upd_texts2.append(temp[0])
            #upd_texts1.append(','.join(temp[0]))
            #upd_texts2.append(','.join(temp[1]))
            #print(upd_texts1, upd_texts2)

        return upd_texts1#, upd_texts2

    def get_distance(self,texts):
        upd_dist1 = []
        upd_dist2 = []

        for idx in texts:
            #print(idx)
            key = idx.replace('a photo of ','').replace('.','')
            #print(key)
            temp = self.foreground_dist[key]
            upd_dist1.append(temp[0])
            upd_dist2.append(temp[1])
        return upd_dist1, upd_dist2

    def forward(self, clip_model, images, labels, eps=0.0001):

        image_features = clip_model.encode_image(images)  # [N1, C]
        #TODO: take the labels and get the list(like background) and tokenize it, send to clip
        #text_features = clip_model.encode_text(clip.tokenize(self.foreground).cuda())  # [N2, C]
        texts = to_text(labels, self.dname)
        #dist1, dist2 = self.get_distance(texts)
        #print(len(dist1))
        foreground_texts1 = self.get_texts_for_image(texts)
        #foreground_texts1, foreground_texts2 = self.get_texts_for_image(texts)
        #print(foreground_texts1)#, foreground_texts2)
        
        text_features1 = clip_model.encode_text(clip.tokenize(foreground_texts1).cuda())  # [N2, C]
        #text_features2 = clip_model.encode_text(clip.tokenize(foreground_texts2).cuda())  # [N2, C]

        # normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        #text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)


        FGISimloss  = SimMaxLoss()
        N, C = image_features.size()
        image_features = image_features.reshape(N, 1, C)
        #print("feat",image_features.shape)
        text_features1 = text_features1.reshape(N, C, 1)
        #print("1",text_features1.shape)
        #text_features2 = text_features2.reshape(N,C,1)

        similarity1 = torch.matmul(image_features, text_features1)
        #print("sim", similarity1.shape)
        #similarity2 = torch.matmul(image_features, text_features2)

        L_FGI_sim1 = FGISimloss(similarity1,1)
        #print("L_FGI", L_FGI_sim1)
        #L_FGI_sim2 = FGISimloss(similarity2,1)

        L_FGI_sim = (L_FGI_sim1)# + L_FGI_sim2)/2

        return L_FGI_sim
 


        '''

        if Change_loss == True:
            FGISimloss  = SimMaxLoss()
            N, C = image_features.size()
            image_features = image_features.reshape(N, 1, C)
            text_features = text_features.reshape(N, C, 1)
            text_features_next =text_features_next.reshape(N,C,1)
            similarity = torch.matmul(image_features, text_features)
            similarity_next = torch.matmul(image_features, text_features_next)
            L_FGI_sim = FGISimloss(similarity,1)
            L_FGI_sim2 = FGISimloss(similarity_next,1)
            #print("yay executing")
            return((L_FGI_sim+L_FGI_sim2)/2)
        else:
            logits_per_image = (image_features @ text_features.t())  # [N1, N2]
            mask = torch.zeros_like(logits_per_image)
            mask = torch.where(logits_per_image > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))
            #print("did not execute")
            return -(torch.log(logits_per_image) * mask).sum()
        '''
