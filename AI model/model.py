import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread


class AIModel:
    def __init__(self, 
                 sys_prompt = '''Your name is Mr.Roberto.
                                    You are a sharp, smart, and friendly human.
                                    You pay close attention to details, stay calm,
                                    and talk naturally without overreacting.'''
                                    ):

        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                    dtype = torch.bfloat16,  #compress model to get better resource usage
                                                    attn_implementation="sdpa",
                                                    device_map="auto")

        self.chat_history = [{'role':'system','content':sys_prompt}]


    def _prepare_inputs(self, prompt):

        self.chat_history.append({"role": "user", "content": prompt})

        if len(self.chat_history) > 7:
            self.chat_history[:] = [self.chat_history[0]] + self.chat_history[-6:]

        return self.tokenizer.apply_chat_template(
            self.chat_history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)



    def generate_response(self, prompt):
        inputs = self._prepare_inputs(prompt)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs,  #type:ignore
                                        max_new_tokens=500,
                                        temperature = .6,             #controls the creativity of the model  
                                        top_p =.9,                    #get the most 90% related to the subject and ignore others 
                                        #   top_k = 50,                   # same as top_p but don't use the percentage
                                        repetition_penalty = 1.1,      # to reduce repetition of words
                                            )


        output_message = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                                            skip_special_tokens = True)


        self.chat_history.append({'role':'assistant','content':output_message})

        return output_message



    def generate_stream_response(self, prompt):
        inputs = self._prepare_inputs(prompt)

        # == this section is how to stream text into UI == 
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            **inputs,  #type:ignore
            max_new_tokens = 500,
            temperature = .6,
            top_p = .9, 
            #top_k = 50,                   
            repetition_penalty = 1.1,      
            streamer = streamer
            )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs) #type:ignore
        thread.start()

        full_response = ""
        for new_text in streamer:
            full_response += new_text
            yield new_text

        self.chat_history.append({'role':'assistant', 'content': full_response})