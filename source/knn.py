import json
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import faiss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

class SentRetriever(object):
    def __init__(self,input_path,output_path):
        self.input_sents_file = input_path
        self.output_sents_file = output_path
        self.batch_size = 64

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained("/data2/private/wyl/models/hf-simcse-bert-base-uncased")
        self.model = AutoModel.from_pretrained("/data2/private/wyl/models/hf-simcse-bert-base-uncased").to(self.device)
        self.model.eval()

        self.query_model = AutoModel.from_pretrained("/data2/private/wyl/models/hf-simcse-bert-base-uncased").to(self.device)
        self.query_tokenizer = AutoTokenizer.from_pretrained("/data2/private/wyl/models/hf-simcse-bert-base-uncased")
        self.query_model.eval()

        if os.path.isfile(self.output_sents_file):
            self.sent_features, self.sent_set, self.status_set, self.passage_set, self.cot_set = self.load_sent_features(self.output_sents_file)
            print(f"{len(self.sent_set)} sents loaded from {self.output_sents_file}")
        else:
            self.sent_set, self.status_set, self.passage_set, self.cot_set = self.load_sents()
            self.sent_features = self.build_sent_features()
            self.save_sent_features(self.sent_features, self.sent_set, self.status_set, self.passage_set, self.cot_set, self.output_sents_file)
    
    def load_sents(self):
        i_set = []
        passage_set=[]
        status_set=[]
        cot_set=[]
        test_file = open(self.input_sents_file, 'r', encoding='utf-8')
        test_data = json.load(test_file)
        for element in test_data:
            if 'Question' in element.keys():
                if element['status']=='same':
                    continue
                i_set.append(element['Question'])
                status_set.append(element['status'])
                passage_set.append([])
                cot_set.append([])

        print(f"Loading {len(i_set)} sents in total.")
        return i_set,status_set,passage_set,cot_set
    
    def build_sent_features(self):
        print(f"Build features for {len(self.sent_set)} sents...")
        batch_size, counter = self.batch_size, 0
        batch_text = []
        all_i_features = []
        for i_n in tqdm(self.sent_set):
            counter += 1
            batch_text.append(i_n)
            if counter % batch_size == 0 or counter >= len(self.sent_set):
                with torch.no_grad():
                    i_input = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                    i_feature = self.model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
                i_feature /= i_feature.norm(dim=-1, keepdim=True)
                all_i_features.append(i_feature.squeeze().to('cpu'))
                batch_text = []
        returned_text_features = torch.cat(all_i_features)
        return returned_text_features

    def save_sent_features(self, sent_feats, sent_names, sent_status, sent_passages, sent_cots, path_to_save):
        assert len(sent_feats) == len(sent_names)
        print(f"Save {len(sent_names)} sent features at {path_to_save}...")
        torch.save({'sent_feats':sent_feats, 'sent_names':sent_names, 'sent_status':sent_status, 'sent_passages':sent_passages, 'sent_cots':sent_cots}, path_to_save)
        print(f"Done.")
    
    def load_sent_features(self, path_to_save):
        print(f"Load sent features from {path_to_save}...")
        checkpoint = torch.load(path_to_save)
        return checkpoint['sent_feats'], checkpoint['sent_names'], checkpoint['sent_status'],checkpoint['sent_passages'], checkpoint['sent_cots']


    def get_text_features(self, text):
        self.query_model.eval()
        with torch.no_grad():
            i_input = self.query_tokenizer(text,return_tensors="pt").to(self.device)
            text_features = self.query_model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def setup_faiss(self):
        self.index = faiss.IndexFlatIP(768)   # BERT-base dimension
        self.index.add(self.sent_features.numpy())    # 

    def faiss_retrieve(self, text, topk=5):
        text_f = self.get_text_features(text)
        D, I = self.index.search(text_f.cpu().numpy(), topk)  
        return D, I

def run_gpt3(test_path, output_path, topk):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        question = test_case['Question'].strip()
        
        train_skr_prop = [47, 62, 740]
        ir_not = ir = 0
        D,I = sentRetriever.faiss_retrieve(question, topk) 
        for idx_ in I[0]:
            if sentRetriever.status_set[idx_].strip()=='ir better': 
                ir+=1
            elif sentRetriever.status_set[idx_].strip()=='ir worse':
                ir_not+=1
            else:
                pass
        if train_skr_prop[0]<train_skr_prop[1]:
            if ir_not > ir and (ir_not-ir)>= int((train_skr_prop[1]-train_skr_prop[0])*topk/sum(train_skr_prop[:3])):
                test_case['cot_ir_or_not'] = 'No'
            else:
                test_case['cot_ir_or_not'] = 'Yes'
        else:
            if ir > ir_not and (ir-ir_not)>= int((train_skr_prop[0]-train_skr_prop[1])*topk/sum(train_skr_prop[:3])):
                test_case['cot_ir_or_not'] = 'Yes'    
            else:            
                test_case['cot_ir_or_not'] = 'No' 

    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    input_path='./train_skr.json'

    output_path='./features/train_skr.pt'
    sentRetriever = SentRetriever(input_path, output_path)
    sentRetriever.setup_faiss()


    test_path = '../temporal/dev.json'
    output_path = './dev_skr_knn.json'
    for i in [5,6,7,8,9,10]:
        run_gpt3(test_path, output_path + '_knn_' + str(i), i)
    

