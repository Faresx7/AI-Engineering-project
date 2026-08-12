from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

class AIModel:
    def __init__(self, sys_prompt = '''Your name is Mr.Roberto.
                                    You are a sharp, smart, and friendly human.
                                    You pay close attention to details, stay calm,
                                    and talk naturally without overreacting.'''):

        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                    torch_dtype = torch.bfloat16,  #compress model to get better resource usage
                                                    attn_implementation="sdpa",
                                                    device_map="auto")

        self.chat_history = [{'role':'system','content':sys_prompt}]

    def generate(self, prompt):
        self.chat_history.append({"role": "user","content": prompt})

        if len(self.chat_history) > 7:
            self.chat_history[:] = [self.chat_history[0]] + self.chat_history[-6:] #[:] to avoid error unbound -> function thinks this is a private variable

        message = self.chat_history
        
        inputs = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                    ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs,  #type:ignore
                                    max_new_tokens=500,
                                    temperature = .6,             #controls the creativity of the model  
                                    top_p =.9,                    #get the most 90% related to the subject and ignore others 
                                    #   top_k = 50,                   # same as top_p but don't use the percentage
                                    repetition_penalty = 1.1,      # to reduce repetition of words
                                    )


        self.output_message = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                                            skip_special_tokens = True)

        self.chat_history.append({'role':'assistant','content':self.output_message})

        return self.output_message



#* to do later:
# enable streaming to show output word by word

# streamer = streamer
# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)