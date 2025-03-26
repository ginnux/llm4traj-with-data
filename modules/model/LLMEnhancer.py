import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model 
from modules.model.base import *

import os

class LLMEnhancer(Encoder):
    def __init__(self, output_size, model_path, sampler, useLoRA=False):
        super().__init__(sampler, f"LLMEnhancer_out{output_size}") 
        self.useLoRA = useLoRA
        self.model_path = model_path
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.city = 'Chengdu'

        
        self.llm = AutoModelForCausalLM.from_pretrained(os.path.join(os.environ['MODEL_PATH'], self.model_path)).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.environ['MODEL_PATH'], self.model_path), padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = self.llm.config.hidden_size

        if useLoRA:
            self.lora_config = LoraConfig(r=16,
                            lora_alpha=32,
                            target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                            lora_dropout=0.05,
                            bias="none",
                            task_type="CAUSAL_LM")
            self.llm = get_peft_model(self.llm, self.lora_config)
        else:
            for param in self.llm.parameters():
                param.requires_grad = False

        # 输出层
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)

    def process_location(self, location, valid_lens):
        B, L, H = location.size()
        assert H == 2, "Location should have 2 dimensions."
        location_list = []
        for b in range(B):
            cut_trip = location[b][:valid_lens[b]].cpu().numpy().tolist()
            round_trip = [[round(num,6) for num in row] for row in cut_trip]
            location_list.append(str(round_trip))
        return location_list
    
    def generate_input(self, location_list):
        # 构建批量的消息列表
        # messages_batch = [
        #     [
        #         {"role": "system", "content":'You are a helpful assist' },
        #         {"role": "user", "content": prompt + location},
        #     ] for location in location_list
        # ]
        input_texts = []
        for location in location_list:
            input_text = "This is a trajectory representation learning task, you need to find the most appropriate word vector in your word vector space to represent this trajectory based on the sequence of trajectory points and additional textual information to the trajectory data." \
                f"Extra Data: The trajectory is record in city {self.city}" \
                f"Trajectory Points Series: {location}." 
            input_texts.append(input_text)
        return input_texts
        

    def forward(self, trips, valid_lens, **kwargs):
        trips = trips[:,:,3:5]
        location_list = self.process_location(trips, valid_lens)
        input_texts = self.generate_input(location_list)
        inputs_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
        outputs = self.llm.generate(
            **inputs_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        logits = outputs.hidden_states[-1][-1]

        # outputs = self.llm(
        #         inputs_ids=inputs_ids,
        #         output_hidden_states=True
        #     )
        # logits = outputs.hidden_states[-1]

        logits = self.fc(logits.squeeze(1))
        
        return logits

if __name__ == "__main__":
    model = LLMEnhancer(10, "meta-llama/Llama-3.2-1B-Instruct", useLoRA=False)
    trips = torch.randn(2, 120, 2).to(model.device)
    valid_lens = torch.randint(0, 121, (2,)).to(model.device)
    print(model(trips, valid_lens).shape)

