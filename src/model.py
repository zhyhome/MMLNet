from transformers import CLIPModel,BertConfig
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import numpy as np

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class SupConLoss2(nn.Module):
    def __init__(self, temperature=0.07, eps=0.5, t=1.00):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.eps = torch.tensor(eps)
        self.t = t

    def forward(self, features, labels=None, mask=None):
        device = features.device
        batch_size = features.shape[0]

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_feature = features

        anchor_dot_contrast = torch.matmul(anchor_feature, anchor_feature.T) / self.temperature

        logits = anchor_dot_contrast - torch.max(anchor_dot_contrast, dim=1, keepdim=True)[0]

        similarity = torch.matmul(anchor_feature, anchor_feature.T)

        mask_same_label = mask - torch.eye(batch_size).to(device) 
        mask_diff_label = 1 - mask

        for i in range(batch_size):
            sample = similarity[i]
            index1 = mask_same_label[i] != 0
            index2 = mask_diff_label[i] != 0

            sample[index1] = 1.0 - torch.softmax(sample[index1], 0).clamp(min=1e-8, max=1.0 - 1e-8)
            sample[index2] = 1.0 + torch.softmax(sample[index2], 0).clamp(min=1e-8, max=1.0 - 1e-8)

        w_same = similarity * mask_same_label / self.t
        w_diff = similarity * mask_diff_label / self.t
        logits_mask = w_same + w_diff

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mask_sum = mask_same_label.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.tensor(1.0, device=device), mask_sum) 

        mean_log_prob_pos = - (log_prob * w_same).sum(1) / mask_sum

        loss = mean_log_prob_pos.sum() / batch_size
        return loss




class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MMLNet(nn.Module):
    def __init__(self, args):
        super(MMLNet, self).__init__()
        self.model = CLIPModel.from_pretrained("/root/autodl-tmp/clip-vit-large-patch14")
        # self.model = ChineseCLIPModel.from_pretrained("/root/autodl-tmp/chinese-clip-vit-large-patch14")
            
        self.config = BertConfig.from_pretrained("/root/autodl-tmp/bert-base-uncased")
        #self.config.hidden_size = 768
        self.config.hidden_size = self.model.config.projection_dim
        self.config.num_attention_heads = 8
        self.dropout_rate = args.dropout_rate
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)
        
        self.ratio = 0.2
        self.img_view_adapter = Adapter(args.image_size, 4).to(self.model.dtype)
        self.text_view_adapter = Adapter(args.text_size, 4).to(self.model.dtype)
        
        self.cl_criterion = SupConLoss2(temperature=0.07,t=0.8)
        self.cl_criterion_image = SupConLoss2(temperature=0.07,t=0.8)
        self.cl_criterion_text = SupConLoss2(temperature=0.07,t=0.8)
        
        self.projection_fuse=nn.Sequential(nn.Linear(args.text_size, args.text_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.text_size, args.text_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.text_size, 128))
        
        self.projection_text=nn.Sequential(nn.Linear(args.text_size, args.text_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.text_size, args.text_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.text_size, 128))
        
        self.projection_image=nn.Sequential(nn.Linear(args.image_size, args.image_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.image_size, args.image_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.image_size, 128))
        
        # weibo
        self.alpha_fuse = nn.Parameter(torch.tensor(1.0))
        self.alpha_text = nn.Parameter(torch.tensor(1.0))
        self.alpha_image = nn.Parameter(torch.tensor(1.0))
        
        self.beta_cl = nn.Parameter(torch.tensor(0.2))

        
        
    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_feature = text_features[:, 0, :] 
        image_feature = image_features[:, 0, :] 

        text_features_ada = self.text_view_adapter(text_feature)
        image_feature_ada = self.img_view_adapter(image_feature)

        text_feature = self.ratio  * text_features_ada + (1 - self.ratio ) * text_feature
        image_feature = self.ratio  * image_feature_ada + (1 - self.ratio ) * image_feature
        
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)
        

        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)
        
        
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        
        image_seq_len = image_embeds.size(1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], image_seq_len).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature
        
        fuse_feature_v = fuse_feature.detach().cpu().numpy()
        labels_v = labels.detach().cpu().numpy()
        
        
        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
        
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        # score = fuse_score + text_score + image_score
        
      
        alpha_fuse = torch.sigmoid(self.alpha_fuse)
        alpha_text = torch.sigmoid(self.alpha_text)
        alpha_image = torch.sigmoid(self.alpha_image)
       
        score = (alpha_fuse * fuse_score +
                 alpha_text * text_score +
                 alpha_image * image_score)
        


        beta_cl = torch.sigmoid(self.beta_cl)

        outputs = (score,fuse_feature_v,labels_v)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            
            cl_fuse_feature = F.normalize(self.projection_fuse(fuse_feature), dim=-1, eps=1e-8)
            cl_text_feature = F.normalize(self.projection_text(text_feature), dim=-1, eps=1e-8)
            cl_image_feature = F.normalize(self.projection_image(image_feature), dim=-1, eps=1e-8)

            cl_loss = self.cl_criterion(cl_fuse_feature,labels)
            cl_loss_image = self.cl_criterion(cl_image_feature,labels)
            cl_loss_text = self.cl_criterion(cl_text_feature,labels)

            # loss = loss_fuse + loss_text + loss_image + 0.2*cl_loss + 0.2*cl_loss_image + 0.2*cl_loss_text

            loss = (alpha_fuse * loss_fuse + 
                    alpha_text * loss_text + 
                    alpha_image * loss_image +
                    # beta_cl * (cl_loss + cl_loss_image + cl_loss_text))
                    beta_cl * (alpha_fuse*cl_loss + alpha_image*cl_loss_image + alpha_text*cl_loss_text))
                    

            outputs = (loss,) + outputs
        return outputs

